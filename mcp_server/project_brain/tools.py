"""Project Brain MCP tools — build and query the shared state substrate.

2 tools:
  build_project_brain — full build from live Ableton session
  get_project_brain_summary — lightweight summary without full rebuild
"""

from __future__ import annotations

from fastmcp import Context

from ..server import mcp
from .builder import build_project_state_from_data
import logging

logger = logging.getLogger(__name__)



def _get_ableton(ctx: Context):
    return ctx.lifespan_context["ableton"]


@mcp.tool()
def build_project_brain(ctx: Context) -> dict:
    """Build a full Project Brain snapshot from the current Ableton session.

    Gathers session info, scenes, clip matrix, track infos with device data,
    builds all five subgraphs (session, arrangement, role, automation,
    capability), and returns the canonical project state.

    This is the primary entry point for engines that need a coherent view
    of the project. Call once at session start, then use scoped refreshes.
    """
    ableton = _get_ableton(ctx)

    # 1. Get session info
    session_info = ableton.send_command("get_session_info")
    tracks = session_info.get("tracks", [])

    # 2. Get scenes info
    scenes = []
    try:
        scenes_resp = ableton.send_command("get_scenes_info")
        scenes = scenes_resp.get("scenes", [])
    except Exception as exc:
        logger.debug("build_project_brain failed: %s", exc)
        scenes = session_info.get("scenes", [])

    # 3. Get clip matrix (scene_matrix)
    clip_matrix = []
    try:
        matrix_resp = ableton.send_command("get_scene_matrix")
        clip_matrix = matrix_resp.get("matrix", [])
    except Exception as exc:
        logger.debug("build_project_brain failed: %s", exc)

    # 4. Gather per-track info with devices
    track_infos = []
    for track in tracks:
        try:
            info = ableton.send_command("get_track_info", {
                "track_index": track["index"],
            })
            track_infos.append(info)
        except Exception as exc:
            logger.debug("build_project_brain failed: %s", exc)
            track_infos.append({
                "index": track.get("index", 0),
                "name": track.get("name", ""),
                "devices": [],
            })

    # 5. Gather arrangement clips per track (legacy path)
    arrangement_clips = {}
    for track in tracks:
        try:
            arr = ableton.send_command("get_arrangement_clips", {
                "track_index": track["index"],
            })
            clips = arr.get("clips", [])
            if clips:
                arrangement_clips[track["index"]] = clips
        except Exception as exc:
            logger.debug("build_project_brain failed: %s", exc)

    # 5b/5c. Single combined grid sweep for notes (role inference) and clip
    # automation envelopes. Previously these were two separate
    # N_tracks x N_scenes loops, each issuing a remote round-trip per slot —
    # including empty slots. We now consult the get_scene_matrix presence grid
    # (already fetched at step 3 as clip_matrix) and skip slots that are
    # positively empty, collapsing both sweeps into one pass over the grid.
    #
    # BUG-E1: section_id must match what build_section_graph_from_scenes emits.
    # The composition engine emits `sec_{i:02d}` using the RAW enumerate index
    # of the scene — it skips unnamed scenes (gap-preserving), so e.g. scenes
    # ["Intro", "", "Verse"] become sections sec_00 and sec_02, not sec_01.
    # Both maps mirror that or keys won't align.
    #
    # clips_scanned is the denominator for coverage_pct (BUG-D2): it counts
    # the slots we actually probed for envelopes. Slots skipped as empty are
    # not counted, so coverage_pct stays "fraction of present clips automated".
    def _slot_is_empty(s_idx: int, t_idx: int) -> bool:
        """True only when the presence grid positively reports no clip.

        Unknown / out-of-range / malformed cells return False so we still
        issue the round-trip (safe fallback — never skip on ambiguity).
        """
        try:
            cell = clip_matrix[s_idx][t_idx]
        except (IndexError, TypeError, KeyError):
            return False
        if not isinstance(cell, dict):
            return False
        if cell.get("has_clip"):
            return False
        return cell.get("state") in ("empty", "missing")

    notes_map: dict[str, dict[int, list[dict]]] = {}
    clip_automation: list[dict] = []
    clips_scanned = 0
    try:
        for scene_idx, scene in enumerate(scenes or []):
            scene_name = str(scene.get("name", "")).strip()
            if not scene_name:
                continue  # mirror _ce_build_sections: unnamed scenes skipped
            section_id = f"sec_{scene_idx:02d}"

            per_track: dict[int, list[dict]] = {}
            for track in tracks:
                t_idx = track.get("index", 0)
                if _slot_is_empty(scene_idx, t_idx):
                    continue  # no clip in this slot — skip both round-trips
                clips_scanned += 1

                # Notes for role inference.
                try:
                    notes_resp = ableton.send_command("get_notes", {
                        "track_index": t_idx,
                        "clip_index": scene_idx,
                    })
                    if isinstance(notes_resp, dict):
                        notes = notes_resp.get("notes", [])
                        if notes:
                            per_track[t_idx] = notes
                except Exception as exc:
                    logger.debug("build_project_brain notes fetch failed: %s", exc)

                # Clip automation envelopes (BUG-E2).
                try:
                    auto_resp = ableton.send_command("get_clip_automation", {
                        "track_index": t_idx,
                        "clip_index": scene_idx,
                    })
                except Exception as exc:
                    # No clip in slot, or remote script rejected — skip
                    logger.debug("build_project_brain automation skip: %s", exc)
                    auto_resp = None
                if isinstance(auto_resp, dict):
                    for env in (auto_resp.get("envelopes") or []):
                        clip_automation.append({
                            "section_id": section_id,
                            "track_index": t_idx,
                            "track_name": track.get("name", ""),
                            "clip_index": scene_idx,
                            "parameter_name": env.get("parameter_name", ""),
                            "parameter_type": env.get("parameter_type", ""),
                            "device_name": env.get("device_name"),
                        })

            if per_track:
                notes_map[section_id] = per_track
    except Exception as exc:
        logger.debug("build_project_brain grid sweep failed: %s", exc)
        # Overall failure: empty maps, degrade to "all tracks active" fallback
        notes_map = {}
        clip_automation = []
        clips_scanned = 0

    # 6. Probe capabilities (direct SpectralCache access, not TCP)
    analyzer_ok = False
    analyzer_fresh = False
    flucoma_ok = False
    try:
        spectral = ctx.lifespan_context.get("spectral")
        if spectral:
            analyzer_ok = spectral.is_connected
            if analyzer_ok:
                snap = spectral.get("spectrum")
                analyzer_fresh = snap is not None
            # Check FluCoMa by looking for any FluCoMa stream data
            for key in ("spectral_shape", "mel_bands", "chroma", "onset", "novelty", "loudness"):
                if spectral.get(key) is not None:
                    flucoma_ok = True
                    break
    except Exception as exc:
        logger.debug("build_project_brain failed: %s", exc)

    # 7. Build state
    state = build_project_state_from_data(
        session_info=session_info,
        scenes=scenes if scenes and clip_matrix else None,
        clip_matrix=clip_matrix if clip_matrix else None,
        track_infos=track_infos if track_infos else None,
        notes_map=notes_map if notes_map else None,
        arrangement_clips=arrangement_clips if arrangement_clips else None,
        clip_automation=clip_automation if clip_automation else None,
        clips_scanned=clips_scanned,
        analyzer_ok=analyzer_ok,
        flucoma_ok=flucoma_ok,
        session_ok=True,
        analyzer_fresh=analyzer_fresh,
        previous_revision=0,
    )

    return state.to_dict()


@mcp.tool()
def get_project_brain_summary(ctx: Context) -> dict:
    """Get a lightweight Project Brain summary — track count, section count, stale status.

    Faster than build_project_brain when you just need an overview.
    Builds session graph only, skips deep inference.
    """
    ableton = _get_ableton(ctx)
    session_info = ableton.send_command("get_session_info")

    state = build_project_state_from_data(
        session_info=session_info,
        previous_revision=0,
    )

    return {
        "project_id": state.project_id,
        "revision": state.revision,
        "track_count": len(state.session_graph.tracks),
        "return_track_count": len(state.session_graph.return_tracks),
        "scene_count": len(state.session_graph.scenes),
        "section_count": len(state.arrangement_graph.sections),
        "role_count": len(state.role_graph.roles),
        "automated_param_count": len(state.automation_graph.automated_params),
        "automation_coverage_pct": round(state.automation_graph.coverage_pct, 3),
        "tempo": state.session_graph.tempo,
        "time_signature": state.session_graph.time_signature,
        "is_stale": state.is_stale(),
    }

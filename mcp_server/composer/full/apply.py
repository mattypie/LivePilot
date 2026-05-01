"""Full compose Phase-3 executor — applies engine-generated plan to live session."""

from __future__ import annotations

import logging
import re as _re
import time

from fastmcp import Context

logger = logging.getLogger(__name__)

# Magic ratios for "smart" mode — common polyrhythmic + half/double-time
# relationships that produce musically interesting results when a loop
# plays un-warped against a project at a different tempo.
_MEANINGFUL_TEMPO_RATIOS: tuple[float, ...] = (
    0.5,    # half-time (project is 2× source)
    0.667,  # 2:3 polyrhythm (project is 1.5× source)
    0.75,   # 3:4 cross-rhythm (project is 1.333× source)
    0.8,    # 4:5
    1.25,   # 5:4
    1.333,  # 4:3 cross-rhythm (source is 1.333× project)
    1.5,    # 3:2 polyrhythm
    2.0,    # double-time
)
_MEANINGFUL_RATIO_TOLERANCE = 0.02  # ±2%

# BPM hint pattern — matches the same naming conventions as
# `_LOOP_FILENAME_RE` in `_analyzer_engine/sample.py`. Splice files use
# `_125_` or `_125bpm` or `125 BPM` style.
_BPM_FROM_FILENAME_RE = _re.compile(
    r"(?:_|\b)(\d{2,3})\s*(?:_|bpm|\b)",
    _re.IGNORECASE,
)


def _extract_bpm_from_filename(file_path: str) -> int | None:
    """Pull a plausible BPM (60-200) from a sample's filename.

    Splice files embed BPM in the basename: `lfh_drums_125_hubble.wav`
    → 125. Returns None if no plausible BPM hint exists (one-shots,
    tonal samples named by key only, etc.). The 60-200 range filters
    out catalog IDs that happen to be 3-digit numbers.
    """
    if not file_path:
        return None
    import os as _os
    stem = _os.path.splitext(_os.path.basename(file_path))[0]
    for match in _BPM_FROM_FILENAME_RE.findall(stem):
        try:
            n = int(match)
        except (ValueError, TypeError):
            continue
        if 60 <= n <= 200:
            return n
    return None


def _is_meaningful_ratio(
    source_bpm: int | float | None,
    project_bpm: int | float | None,
    tolerance: float = _MEANINGFUL_RATIO_TOLERANCE,
) -> bool:
    """Return True when source/project BPM ratio is in the magic set ±tol.

    Used by 'smart' warp strategy to decide when to leave a loop
    un-warped (because the tempo mismatch creates interesting chopping)
    versus warping it to project tempo (the production-safe default).

    Defensive on None / 0 inputs — returns False rather than blowing up
    on missing BPM data.
    """
    if not source_bpm or not project_bpm:
        return False
    try:
        ratio = float(source_bpm) / float(project_bpm)
    except (ZeroDivisionError, ValueError, TypeError):
        return False
    for magic in _MEANINGFUL_TEMPO_RATIOS:
        if abs(ratio - magic) / magic <= tolerance:
            return True
    return False


# Roles whose layers should ALWAYS warp regardless of ratio. Tonal /
# harmonic content sounds wrong when un-warped — only drums benefit
# from intentional chopping.
_TONAL_ROLES_ALWAYS_WARP: frozenset[str] = frozenset({
    "pad", "bass", "lead", "vocal", "texture", "fx",
})


def _decide_warp_loops(
    role: str,
    file_path: str,
    project_tempo: int | float | None,
    strategy: str,
) -> bool:
    """Decide whether to warp this loop based on strategy + role + ratio.

    Strategy semantics:
      - "always" → always True (production-safe default)
      - "chop"   → always False (creative chopping mode)
      - "smart"  → True for tonal roles; for drum/perc, False if the
                   source/project BPM ratio lands on a magic ratio.

    Returns the boolean to pass as `warp_loops` to load_sample_to_simpler.
    """
    s = (strategy or "always").lower().strip()
    if s == "chop":
        return False
    if s == "always":
        return True
    if s == "smart":
        # Tonal roles always warp — chopping a pad sounds glitchy bad
        if (role or "").lower() in _TONAL_ROLES_ALWAYS_WARP:
            return True
        # Drum/perc with no project tempo → can't compute ratio → warp
        if not project_tempo:
            return True
        source_bpm = _extract_bpm_from_filename(file_path)
        if not source_bpm:
            return True  # no BPM hint → can't be sure → safe default
        # Meaningful ratio → leave un-warped for creative chopping
        if _is_meaningful_ratio(source_bpm, project_tempo):
            return False
        return True
    # Unknown strategy → default to always
    return True


def _resolve_from_step(value, step_results: dict):
    """Recursively substitute ``$from_step`` placeholders inside plan params.

    The plan emits cross-step references for things like the device_index
    of a freshly inserted device:

        {"$from_step": "layer_0_dev_0", "path": "device_index"}

    The walker captures every step's response keyed by its ``step_id``.
    This helper walks the params tree and replaces those placeholders
    with the actual values before dispatching the call.
    """
    if isinstance(value, dict):
        if "$from_step" in value:
            ref_id = value["$from_step"]
            path = str(value.get("path", "") or "")
            if ref_id not in step_results:
                raise ValueError(
                    f"$from_step references unknown step '{ref_id}' "
                    f"(known: {sorted(step_results.keys())})"
                )
            current = step_results[ref_id]
            if path:
                for key in path.split("."):
                    if not key:
                        continue
                    if not isinstance(current, dict) or key not in current:
                        raise ValueError(
                            f"$from_step path '{path}' not found in step "
                            f"'{ref_id}' result keys={list(current.keys()) if isinstance(current, dict) else type(current).__name__}"
                        )
                    current = current[key]
            return current
        return {k: _resolve_from_step(v, step_results) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_from_step(v, step_results) for v in value]
    return value


# Plan tools that aren't direct Remote-Script TCP commands — they need
# special dispatch (either a Python function call or a multi-step bridge
# routine like `load_sample_to_simpler` which itself does multiple TCP
# operations under the hood).
_FULL_PLAN_TCP_TOOLS = {
    "set_tempo",
    "create_midi_track",
    "create_audio_track",
    "create_return_track",
    "create_scene",
    "set_track_name",
    "set_track_volume",
    "set_track_pan",
    "set_track_send",
    "insert_device",
    "set_device_parameter",
    "create_clip",
    "add_notes",
    "create_arrangement_clip",
    "set_clip_color",
    "set_track_color",
    "set_clip_name",
    "set_clip_loop",
}


async def apply_full_plan(
    ctx: Context,
    plan_response: dict,
    warp_strategy: str = "always",
) -> dict:
    """Phase-3 full mode: server-side execute the planner's tool sequence.

    Pre-flight handles the same fresh-project cleanup fast mode does
    (BUG-FULL-MODE-4): detects default tracks, deletes them down to one
    survivor, loads the LivePilot Analyzer on master, sets the project
    tempo. Then walks the plan's `plan` array sequentially, resolving
    `$from_step` references against accumulated step results. After the
    walk, deletes the leftover default track if it's still empty
    (BUG-FULL-MODE-5).

    `warp_strategy` (BUG-FULL-MODE-12, 2026-05-01) controls per-step
    Simpler warping behavior:
      - "always" (default): every loop warps to project tempo
      - "smart": tonal layers always warp; drum/perc loops un-warped
        when source/project ratio is musically meaningful (creates
        creative tempo-mismatch chopping — J Dilla / Madlib territory)
      - "chop": no warping anywhere (pure creative chopping)
    """
    from .. import fast as fast_compose

    started = time.time()
    ableton = ctx.lifespan_context.get("ableton") if hasattr(ctx, "lifespan_context") else None
    if ableton is None:
        return {"error": "Ableton connection not available", "phase": "apply"}

    plan_steps = plan_response.get("plan") or []
    if not plan_steps:
        return {"error": "plan.plan is empty — nothing to apply", "phase": "apply"}

    # ── Pre-flight (Item 4) ─────────────────────────────────────────
    # Mirror fast mode's fresh-project cleanup: load analyzer, detect
    # default tracks, delete all-but-one (Ableton requires ≥1 track).
    fresh_actions: list[str] = []

    # 1. Ensure analyzer on master so load_sample_to_simpler can succeed
    try:
        from ...tools.analyzer import ensure_analyzer_on_master as _ensure_analyzer
        analyzer_resp = _ensure_analyzer(ctx)
        if analyzer_resp.get("status") in ("loaded", "already_loaded"):
            fresh_actions.append("analyzer_loaded_on_master")
    except Exception as exc:
        logger.debug("full apply: ensure_analyzer_on_master failed: %s", exc)

    # 1b. Reconnect M4L UDP bridge (BUG-FULL-MODE-7, 2026-05-01).
    # When the analyzer was just freshly loaded by step 1, its M4L UDP
    # listener may not have registered yet — load_sample_to_simpler's
    # bridge-driven steps (replace_sample, hygiene) will fail with
    # "bridge is not connected" until the listener bootstraps. Forcing a
    # reconnect_bridge call here ensures the bridge is alive before the
    # plan walk reaches any sample-loading steps. Idempotent: returns
    # "already connected" when the bridge is healthy.
    try:
        from ...tools.analyzer import reconnect_bridge as _reconnect_bridge_fn
        bridge_resp = await _reconnect_bridge_fn(ctx)
        if bridge_resp.get("ok"):
            fresh_actions.append("bridge_connected")
    except Exception as exc:
        logger.debug("full apply: reconnect_bridge failed: %s", exc)

    # 2. Detect + clean default tracks
    session = ableton.send_command("get_session_info", {})
    starting_track_count = int(session.get("track_count", 0))

    fresh_project = fast_compose.detect_fresh_project(session)
    if fresh_project:
        candidates: list[int] = []
        for i in range(starting_track_count):
            try:
                ti = ableton.send_command("get_track_info", {"track_index": i})
                if fast_compose.track_is_empty(ti):
                    candidates.append(i)
            except Exception as exc:
                logger.debug("full apply: fresh-check get_track_info(%s) failed: %s", i, exc)

        if len(candidates) == starting_track_count and starting_track_count > 0:
            fresh_actions.append(f"detected_fresh_project_{starting_track_count}_default_tracks")
            # Leave one survivor — Ableton requires ≥1 track at all times
            deletable = sorted(candidates, reverse=True)[:-1]
            deleted = 0
            for idx in deletable:
                try:
                    ableton.send_command("delete_track", {"track_index": idx})
                    deleted += 1
                except Exception as exc:
                    logger.debug("full apply: delete_track(%s) failed: %s", idx, exc)
            if deleted:
                fresh_actions.append(f"deleted_{deleted}_default_tracks_preflight")

    # ── Walk plan steps ────────────────────────────────────────────
    step_results: dict[str, dict] = {}
    step_outcomes: list[dict] = []
    failed_count = 0

    for i, step in enumerate(plan_steps):
        tool_name = (step.get("tool") or "").strip()
        params = step.get("params") or {}
        step_id = step.get("step_id")
        description = step.get("description") or ""
        role = step.get("role")

        # Resolve $from_step refs inside params
        try:
            resolved_params = _resolve_from_step(params, step_results)
        except Exception as exc:
            failed_count += 1
            step_outcomes.append({
                "index": i, "tool": tool_name, "step_id": step_id,
                "description": description, "role": role,
                "ok": False, "error": f"$from_step resolution failed: {exc}",
            })
            continue

        # Dispatch
        result: dict = {}
        ok = True
        err_msg: str | None = None
        try:
            if tool_name == "load_sample_to_simpler":
                # Special-case: this is an MCP tool that wraps multi-step
                # bridge work (verify, replace, hygiene). Call it as a
                # Python function rather than a single TCP command.
                #
                # BUG-FULL-MODE-12: translate warp_strategy → per-step
                # warp_loops bool, based on this layer's role + the
                # source loop's BPM ratio against project tempo.
                project_tempo = (plan_response.get("intent") or {}).get("tempo")
                warp_loops_decision = _decide_warp_loops(
                    role=role or "",
                    file_path=str(resolved_params.get("file_path", "")),
                    project_tempo=project_tempo,
                    strategy=warp_strategy,
                )
                # Don't override an explicit warp_loops in the plan
                # params (lets the planner — or a manual edit — pin a
                # specific layer's warp setting regardless of strategy).
                if "warp_loops" not in resolved_params:
                    resolved_params["warp_loops"] = warp_loops_decision

                from ...tools.analyzer import load_sample_to_simpler as _load_sample
                # The MCP tool is async — await it with the resolved kwargs.
                result = await _load_sample(ctx, **resolved_params)
            elif tool_name in _FULL_PLAN_TCP_TOOLS:
                # Direct Remote-Script TCP command
                result = ableton.send_command(tool_name, resolved_params) or {}
            else:
                # Unknown tool — try generic TCP send (most LivePilot tools
                # have a 1:1 Remote-Script handler with the same name).
                result = ableton.send_command(tool_name, resolved_params) or {}
        except Exception as exc:
            ok = False
            err_msg = str(exc)
            failed_count += 1
            logger.debug("full apply step %s (%s) failed: %s", i, tool_name, exc)

        if step_id and ok:
            step_results[step_id] = result if isinstance(result, dict) else {}

        step_outcomes.append({
            "index": i,
            "tool": tool_name,
            "step_id": step_id,
            "description": description,
            "role": role,
            "ok": ok,
            "error": err_msg,
        })

    # ── Post-flight cleanup (Item 5) ───────────────────────────────
    # BUG-FULL-MODE-8 (2026-05-01): the original implementation only
    # checked tracks[0] for a default-name leftover. That worked for
    # fast mode where new tracks are appended at the end (so the
    # survivor stays at index 0), but full mode's planner creates
    # tracks at SPECIFIC indices (0, 1, 2, 3, 4...) which pushes the
    # survivor to index N. Fix: scan ALL tracks and prune every empty
    # default-named one. Walk highest-to-lowest so deletions don't
    # invalidate the indices below.
    final_cleanup_actions: list[str] = []
    try:
        post_session = ableton.send_command("get_session_info", {})
        tracks = post_session.get("tracks", []) or []
        if tracks and len(tracks) > 1:
            default_indices: list[int] = []
            for i, t in enumerate(tracks):
                if fast_compose.is_default_track_name(t.get("name", "")):
                    try:
                        ti = ableton.send_command("get_track_info", {"track_index": i})
                        if fast_compose.track_is_empty(ti):
                            default_indices.append(i)
                    except Exception as exc:
                        logger.debug("full apply: cleanup get_track_info(%s) failed: %s", i, exc)
            # Delete highest-to-lowest so earlier deletions don't shift
            # the indices we still need to delete.
            for idx in sorted(default_indices, reverse=True):
                # Don't delete if we'd end up with zero tracks
                if len(tracks) - len(final_cleanup_actions) <= 1:
                    break
                try:
                    ableton.send_command("delete_track", {"track_index": idx})
                    final_cleanup_actions.append(f"deleted_leftover_default_track_at_{idx}")
                except Exception as exc:
                    logger.debug("full apply: final cleanup delete_track(%s) failed: %s", idx, exc)
    except Exception as exc:
        logger.debug("full apply: post-session read failed: %s", exc)

    duration_ms = int((time.time() - started) * 1000)
    return {
        "phase": "apply",
        "mode": "full",
        "steps_executed": len(step_outcomes),
        "steps_failed": failed_count,
        "step_outcomes": step_outcomes,
        "fresh_project_actions": fresh_actions,
        "final_cleanup_actions": final_cleanup_actions,
        "duration_ms": duration_ms,
        "summary": (
            f"{len(step_outcomes)} steps walked, "
            f"{failed_count} failed, "
            f"{len(fresh_actions)} pre-flight action(s), "
            f"{len(final_cleanup_actions)} cleanup action(s)"
        ),
    }

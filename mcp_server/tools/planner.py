"""Planner MCP tools — loop-to-song arrangement planning.

2 tools that connect the planner engine (_planner_engine.py) to the
live Ableton session.

  plan_arrangement — transform a loop into a full arrangement blueprint
  get_emotional_arc — (in research.py, shares composition data)
"""

from __future__ import annotations

import json
from typing import Optional

from fastmcp import Context

from ..server import mcp
from . import _composition_engine as comp_engine
from . import _planner_engine as planner_engine
import logging

logger = logging.getLogger(__name__)


def _get_ableton(ctx: Context):
    return ctx.lifespan_context["ableton"]


def _resolve_section_template(sections):
    """Build the (SectionType, energy, density, bars) template for the planner.

    Honors a caller-supplied ``sections`` list (entries shaped as
    ``[type, energy, density, bars]`` or the dict form); otherwise falls back
    to a generic genre-neutral arc that starts INTRO and ends OUTRO. The v1.24
    refactor removed the built-in per-genre STYLE_TEMPLATES
    (vocabulary-not-form) — the framework supplies only a neutral default
    skeleton so plan_arrangement always returns a plan instead of crashing on
    the now-mandatory section_template argument.
    """
    ST = comp_engine.SectionType
    if sections:
        template = []
        for entry in sections:
            if isinstance(entry, dict):
                stype = entry.get("type") or entry.get("section_type") or "verse"
                energy = float(entry.get("energy", entry.get("energy_target", 0.5)))
                density = float(entry.get("density", entry.get("density_target", 0.5)))
                bars = max(1, int(entry.get("bars", 8)))
            else:
                stype, energy, density, bars = entry
                # Clamp bars>=1: a 0/negative bar count makes the engine's
                # template_bars sum 0 -> ZeroDivisionError on scale_factor.
                energy, density, bars = float(energy), float(density), max(1, int(bars))
            if not isinstance(stype, ST):
                try:
                    stype = ST(str(stype).lower())
                except ValueError:
                    stype = ST.VERSE
            template.append((stype, energy, density, bars))
        if template:
            return template
    # Generic genre-neutral default arc (INTRO … OUTRO).
    return [
        (ST.INTRO, 0.3, 0.3, 8),
        (ST.BUILD, 0.6, 0.6, 8),
        (ST.DROP, 1.0, 0.9, 16),
        (ST.BREAKDOWN, 0.5, 0.4, 8),
        (ST.BUILD, 0.7, 0.7, 8),
        (ST.DROP, 1.0, 1.0, 16),
        (ST.OUTRO, 0.3, 0.2, 8),
    ]


@mcp.tool()
def plan_arrangement(
    ctx: Context,
    target_bars: int = 128,
    style: str = "electronic",
    sections: Optional[list] = None,
) -> dict:
    """Transform the current loop/session into a full arrangement blueprint.

    Analyzes the existing tracks and their roles, then proposes:
    - Section sequence (intro → verse → build → drop → etc.)
    - Element reveal order (what enters/exits when)
    - Gesture automation suggestions for transitions
    - Orchestration plan (which tracks play in which sections)

    target_bars: desired total arrangement length (default: 128 bars)
    style: free-text style label (e.g. "electronic", "ambient") — recorded as a
        hint on the result. The framework no longer hardcodes genre→form
        templates (vocabulary-not-form, v1.24); supply explicit form via
        `sections` if desired.
    sections: optional explicit form — a list of
        [section_type, energy_target, density_target, bars] entries (or the
        dict form {"type","energy","density","bars"}). When omitted a generic
        genre-neutral arc (INTRO…OUTRO) is used so the tool always returns a
        plan.

    Returns: full ArrangementPlan with actionable section-by-section instructions.
    """
    ableton = _get_ableton(ctx)
    # Capture the caller's requested form before the local `sections` (the
    # built section graph) shadows the parameter name below.
    requested_sections = sections

    # 1. Get session info
    session = ableton.send_command("get_session_info")
    scenes = session.get("scenes", [])
    tracks = session.get("tracks", [])
    track_count = session.get("track_count", 0)

    # 2. Build section graph (to analyze current state)
    from .composition import _build_clip_matrix
    clip_matrix = _build_clip_matrix(ableton, len(scenes), track_count)
    sections = comp_engine.build_section_graph_from_scenes(scenes, clip_matrix, track_count)

    # 3. Get track info for role inference
    track_data = []
    notes_map: dict[str, dict[int, list]] = {}

    for track in tracks:
        t_idx = track["index"]
        try:
            ti = ableton.send_command("get_track_info", {"track_index": t_idx})
            track_data.append(ti)
        except Exception as exc:
            logger.debug("plan_arrangement failed: %s", exc)
            track_data.append({"index": t_idx, "name": track.get("name", ""), "devices": []})

    for section in sections:
        notes_map[section.section_id] = {}
        for t_idx in section.tracks_active:
            notes_map[section.section_id][t_idx] = []

    # 4. Build role graph
    roles = comp_engine.build_role_graph(sections, track_data, notes_map)

    # 5. Analyze loop identity
    loop_identity = planner_engine.analyze_loop_identity(roles, sections)

    # 6. Plan arrangement. The v1.24 refactor made section_template mandatory
    # (STYLE_TEMPLATES removed); supply the caller's form or a neutral default.
    # Malformed `sections` (wrong arity, non-numeric energy/density) would raise
    # a raw ValueError/TypeError; convert to the file's structured-error form.
    try:
        section_template = _resolve_section_template(requested_sections)
    except (ValueError, TypeError) as exc:
        return {"error": f"Invalid sections entry: {exc}", "code": "INVALID_PARAM"}
    plan = planner_engine.plan_arrangement_from_loop(
        loop_identity,
        target_duration_bars=target_bars,
        style=style,
        section_template=section_template,
    )

    # Add section-level sample role hints
    planner_engine.add_sample_hints(plan)

    result = plan.to_dict()
    result["loop_identity"] = loop_identity.to_dict()
    result["style"] = style
    return result

# ── transform_section (Round 4) ─────────────────────────────────────


@mcp.tool()
def transform_section(
    ctx: Context,
    transformation: str,
    section_index: int = -1,
    bars: int = 8,
) -> dict:
    """Apply a structural transformation to the arrangement.

    Proposes radical structural moves — reorder sections, expand loops,
    compress verbose arrangements, insert bridges. Returns the proposed
    new section graph without modifying the actual session.

    transformation: insert_bridge_before_final_chorus | swap_verse_positions |
                    extend_section | compress_section | insert_breakdown |
                    duplicate_section | remove_section | reverse_section_order |
                    split_section
    section_index: which section to transform (required for targeted operations, -1 = auto)
    bars: how many bars for extend/compress/insert operations

    Returns: before/after section graphs with description and bar delta.
    """
    from . import _form_engine as form_engine

    ableton = _get_ableton(ctx)
    session = ableton.send_command("get_session_info")
    scenes = session.get("scenes", [])
    track_count = session.get("track_count", 0)

    from .composition import _build_clip_matrix

    clip_matrix = _build_clip_matrix(ableton, len(scenes), track_count)
    sections = comp_engine.build_section_graph_from_scenes(scenes, clip_matrix, track_count)

    if not sections:
        return {"error": "No sections detected in the arrangement"}

    target = section_index if section_index >= 0 else None

    try:
        result = form_engine.transform_section_order(
            sections, transformation, target_index=target, bars=bars,
        )
        return result.to_dict()
    except ValueError as e:
        return {"error": str(e)}

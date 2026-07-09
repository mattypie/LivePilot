"""Regression tests for P2-21 — mix/sound-design/transition compilers must
compile RELATIVE volume nudges from a track's current level, not blind
absolute overwrites.

Before the fix, e.g. make_punchier always wrote drums to a flat 0.75 and
pads to a flat 0.25 regardless of what was already there — "make it
punchier" could turn a hot drum bus DOWN. resolvers.find_tracks_by_role
already carried a "volume" key (resolvers.py) that was silently always None
because get_session_info never emitted it (transport.py) — so this also
guards the fallback path for Remote Scripts that predate the volume field.
"""

from __future__ import annotations

import pytest

from mcp_server.semantic_moves import resolvers
from mcp_server.semantic_moves.compiler import compile as compile_move
from mcp_server.semantic_moves.registry import get_move


def _kernel_with_volumes(drum_volume=None, pad_volume=None, bass_volume=None):
    tracks = [
        {"index": 0, "name": "Drums", "volume": drum_volume},
        {"index": 1, "name": "Bass", "volume": bass_volume},
        {"index": 2, "name": "Pad", "volume": pad_volume},
    ]
    return {
        "mode": "explore",
        "session_info": {"tempo": 120, "tracks": tracks},
    }


# ── resolvers.compile_relative_volume ────────────────────────────────────

def test_compile_relative_volume_applies_bounded_delta():
    # A hot drum bus (0.80) pushed +8% must land above where it started,
    # not get slammed down to a flat absolute value.
    target = resolvers.compile_relative_volume(0.80, 8, cap=0.85, fallback=0.75)
    assert target == pytest.approx(0.85)  # capped
    assert target > 0.80


def test_compile_relative_volume_negative_delta_respects_floor():
    target = resolvers.compile_relative_volume(0.20, -10, floor=0.15, fallback=0.25)
    assert target == pytest.approx(0.15)  # floored, not 0.10


def test_compile_relative_volume_falls_back_when_current_is_none():
    """Older Remote Script without the `volume` field in get_session_info
    must not crash the compiler — it degrades to the historical absolute."""
    target = resolvers.compile_relative_volume(None, 8, cap=0.85, fallback=0.75)
    assert target == 0.75


# ── make_punchier: the exact P2-21 scenario ──────────────────────────────

def test_make_punchier_does_not_slam_a_hot_drum_bus_down():
    """A drum bus already at 0.9 must NOT be pulled down to 0.75 — the old
    absolute-write bug. It should end up pushed higher (capped at 0.85 or
    above its current level), never below where it started."""
    kernel = _kernel_with_volumes(drum_volume=0.9, pad_volume=0.5)
    move = get_move("make_punchier")
    plan = compile_move(move, kernel)

    drum_steps = [
        s for s in plan.steps
        if s.tool == "set_track_volume" and s.params.get("track_index") == 0
    ]
    assert len(drum_steps) == 1
    new_volume = drum_steps[0].params["volume"]
    assert new_volume >= 0.9, (
        f"make_punchier pulled a hot drum bus DOWN to {new_volume} — "
        "this is exactly the P2-21 regression"
    )


def test_make_punchier_pushes_drums_up_from_moderate_level():
    kernel = _kernel_with_volumes(drum_volume=0.5, pad_volume=0.5)
    move = get_move("make_punchier")
    plan = compile_move(move, kernel)

    drum_steps = [
        s for s in plan.steps
        if s.tool == "set_track_volume" and s.params.get("track_index") == 0
    ]
    assert drum_steps[0].params["volume"] == pytest.approx(0.58)  # 0.5 + 0.08


def test_make_punchier_pulls_pads_down_but_floors_a_quiet_pad():
    kernel = _kernel_with_volumes(drum_volume=0.6, pad_volume=0.18)
    move = get_move("make_punchier")
    plan = compile_move(move, kernel)

    pad_steps = [
        s for s in plan.steps
        if s.tool == "set_track_volume" and s.params.get("track_index") == 2
    ]
    assert len(pad_steps) == 1
    assert pad_steps[0].params["volume"] == pytest.approx(0.15)  # floored


def test_make_punchier_falls_back_to_absolute_when_volume_key_missing():
    """When the kernel's tracks have no volume info at all (old Remote
    Script), the compiler must still produce a valid plan using the
    historical absolute values — no crash."""
    tracks = [
        {"index": 0, "name": "Drums"},  # no "volume" key at all
        {"index": 1, "name": "Pad"},
    ]
    kernel = {"mode": "explore", "session_info": {"tempo": 120, "tracks": tracks}}
    move = get_move("make_punchier")
    plan = compile_move(move, kernel)

    drum_steps = [
        s for s in plan.steps
        if s.tool == "set_track_volume" and s.params.get("track_index") == 0
    ]
    pad_steps = [
        s for s in plan.steps
        if s.tool == "set_track_volume" and s.params.get("track_index") == 1
    ]
    assert drum_steps[0].params["volume"] == pytest.approx(0.75)
    assert pad_steps[0].params["volume"] == pytest.approx(0.25)


# ── tighten_low_end: bass reduction must not crash toward silence ───────

def test_tighten_low_end_floors_an_already_quiet_bass():
    kernel = _kernel_with_volumes(bass_volume=0.36)
    move = get_move("tighten_low_end")
    plan = compile_move(move, kernel)

    bass_steps = [
        s for s in plan.steps
        if s.tool == "set_track_volume" and s.params.get("track_index") == 1
    ]
    assert len(bass_steps) == 1
    assert bass_steps[0].params["volume"] == pytest.approx(0.35)  # floored


def test_tighten_low_end_reduces_relative_to_current_level():
    kernel = _kernel_with_volumes(bass_volume=0.8)
    move = get_move("tighten_low_end")
    plan = compile_move(move, kernel)

    bass_steps = [
        s for s in plan.steps
        if s.tool == "set_track_volume" and s.params.get("track_index") == 1
    ]
    assert bass_steps[0].params["volume"] == pytest.approx(0.70)  # 0.8 - 0.10

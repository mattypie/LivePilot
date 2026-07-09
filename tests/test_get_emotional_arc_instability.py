"""Test for P2-46: get_emotional_arc's tension formula must use a real
harmonic-instability signal.

Pre-fix, ``HarmonyField.instability`` was never assigned inside
``get_emotional_arc`` (models default 0.0), so the tension formula's
``instability * 0.2`` term was dead weight — every section's instability
contribution was 0.0 regardless of how harmonically stable/unstable the
detected key actually was.

Post-fix, instability is derived from:
  1. detect_key's confidence (low confidence == harmonically ambiguous ==
     unstable) — ``1.0 - confidence``, clamped to [0, 1].
  2. A +0.15 bump when the detected mode differs from the previous
     section's mode (a mode shift is itself destabilizing).

This test pins both signals down with a fully controlled fake session
(single track, always active — so every section has the same
energy/density) and a monkeypatched ``detect_key`` returning fixed
confidence/mode per section, then checks the resulting `tension` values
algebraically.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _fake_ableton():
    session_info = {
        "tempo": 120,
        "track_count": 1,
        "tracks": [{"index": 0, "name": "Lead"}],
        "scenes": [
            {"index": 0, "name": "Intro"},
            {"index": 1, "name": "Build"},
            {"index": 2, "name": "Drop"},
        ],
    }
    # Single track, active (has_clip) in all 3 scene rows — this pins
    # density == energy == 1.0 for every section, so the only variable
    # term left in the tension formula is instability.
    scene_matrix = {
        "matrix": [
            [{"state": "stopped", "has_clip": True}],
            [{"state": "stopped", "has_clip": True}],
            [{"state": "stopped", "has_clip": True}],
        ]
    }
    notes = [{"pitch": 60, "start_time": 0, "duration": 1, "velocity": 90}]

    def send_command(cmd, params=None):
        params = params or {}
        if cmd == "get_session_info":
            return session_info
        if cmd == "get_scene_matrix":
            return scene_matrix
        if cmd == "get_notes":
            return {"notes": notes}
        return {}

    return SimpleNamespace(send_command=send_command)


def _fake_ctx():
    return SimpleNamespace(lifespan_context={"ableton": _fake_ableton()})


def test_instability_derives_from_key_confidence_and_mode_change(monkeypatch):
    from mcp_server.tools import research
    from mcp_server.tools import _theory_engine as theory_engine

    # Section 0: confident major key -> low instability, no prior section
    #            to compare against (no mode-change bump possible).
    # Section 1: same confidence/mode as section 0 -> low instability,
    #            no mode-change bump (mode unchanged).
    # Section 2: low-confidence minor key, mode DIFFERS from section 1's
    #            major -> high base instability + the mode-change bump.
    responses = [
        {"tonic_name": "C", "mode": "major", "confidence": 0.9},
        {"tonic_name": "C", "mode": "major", "confidence": 0.9},
        {"tonic_name": "A", "mode": "minor", "confidence": 0.2},
    ]
    calls = {"n": 0}

    def fake_detect_key(notes, mode_detection=True):
        idx = calls["n"]
        calls["n"] += 1
        return responses[idx]

    monkeypatch.setattr(theory_engine, "detect_key", fake_detect_key)

    ctx = _fake_ctx()
    result = research.get_emotional_arc(ctx)

    assert result["section_count"] == 3
    curve = result["tension_curve"]
    assert len(curve) == 3

    # density == energy == 1.0 for every section (single always-active
    # track), so tension = 1.0*0.5 + 1.0*0.3 + instability*0.2
    #                     = 0.8 + instability*0.2
    for point in curve:
        assert point["energy"] == 1.0
        assert point["density"] == 1.0

    # Section 0: instability = 1.0 - 0.9 = 0.1 (no previous mode to compare)
    assert curve[0]["tension"] == round(0.8 + 0.1 * 0.2, 3)

    # Section 1: same mode as section 0 -> no mode-change bump.
    assert curve[1]["tension"] == round(0.8 + 0.1 * 0.2, 3)

    # Section 2: instability = 1.0 - 0.2 = 0.8, mode changed
    # (major -> minor) vs section 1 -> +0.15 bump -> 0.95.
    assert curve[2]["tension"] == round(0.8 + 0.95 * 0.2, 3)

    # The whole point of the fix: instability must NOT be a dead 0.0
    # contribution across the board — section 2's tension must differ
    # from the low-instability sections.
    assert curve[2]["tension"] > curve[0]["tension"]


def test_instability_falls_back_to_neutral_when_no_key_detected(monkeypatch):
    """When a section has no notes at all (detect_key never runs, hf.key
    stays ""), instability should fall back to the neutral 0.3 used
    elsewhere in this tool, not silently stay 0.0."""
    from mcp_server.tools import research

    ableton = SimpleNamespace(send_command=lambda cmd, params=None: (
        {
            "tempo": 120,
            "track_count": 1,
            "tracks": [{"index": 0, "name": "Lead"}],
            "scenes": [
                {"index": 0, "name": "Intro"},
                {"index": 1, "name": "Build"},
                {"index": 2, "name": "Drop"},
            ],
        } if cmd == "get_session_info" else
        {"matrix": [
            [{"state": "stopped", "has_clip": True}],
            [{"state": "stopped", "has_clip": True}],
            [{"state": "stopped", "has_clip": True}],
        ]} if cmd == "get_scene_matrix" else
        {"notes": []} if cmd == "get_notes" else
        {}
    ))
    ctx = SimpleNamespace(lifespan_context={"ableton": ableton})

    result = research.get_emotional_arc(ctx)
    curve = result["tension_curve"]
    # energy == density == 1.0, no key ever detected -> instability == 0.3
    # (neutral fallback) -> tension = 0.8 + 0.3*0.2 = 0.86 for every section.
    for point in curve:
        assert point["tension"] == round(0.8 + 0.3 * 0.2, 3)

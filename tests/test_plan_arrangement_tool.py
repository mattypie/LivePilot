"""Contract test for the plan_arrangement MCP tool (LIVE#6).

plan_arrangement was 100% dead after the v1.24 vocabulary-not-form refactor:
  1. it referenced planner_engine.VALID_STYLES, deleted by commit 06c0336
     (AttributeError on every call), and
  2. even past that, it called plan_arrangement_from_loop() without a
     section_template, which the refactor made mandatory (ValueError).

No test exercised the tool post-refactor, so the suite stayed green while the
registered tool crashed on every invocation. These tests call the real tool
body against a fake Ableton so the crash paths are covered for good.
"""

from __future__ import annotations

from types import SimpleNamespace

from mcp_server.tools.planner import plan_arrangement


class FakeAbleton:
    """Returns canned responses for the commands plan_arrangement issues."""

    def __init__(self, scenes, tracks):
        self._scenes = scenes
        self._tracks = tracks
        self.calls = []

    def send_command(self, command, params=None):
        self.calls.append((command, params))
        if command == "get_session_info":
            return {
                "scenes": self._scenes,
                "tracks": self._tracks,
                "track_count": len(self._tracks),
            }
        if command == "get_scene_matrix":
            # one cell per (scene, track); mark track 0 as having a clip
            matrix = []
            for _ in self._scenes:
                row = [{"state": "playing" if t["index"] == 0 else "empty"}
                       for t in self._tracks]
                matrix.append(row)
            return {"matrix": matrix}
        if command == "get_track_info":
            idx = (params or {}).get("track_index", 0)
            t = next((x for x in self._tracks if x["index"] == idx), None)
            return {"index": idx, "name": (t or {}).get("name", ""), "devices": []}
        return {}


def _ctx(ableton):
    return SimpleNamespace(lifespan_context={"ableton": ableton})


def _session():
    scenes = [{"index": 0, "name": "Loop"}, {"index": 1, "name": "B"}]
    tracks = [
        {"index": 0, "name": "Kick"},
        {"index": 1, "name": "Bass"},
        {"index": 2, "name": "Lead"},
    ]
    return FakeAbleton(scenes, tracks)


def test_plan_arrangement_does_not_crash_default_style():
    """Regression: no AttributeError (VALID_STYLES) and no ValueError
    (missing section_template) — the tool returns a real plan."""
    result = plan_arrangement(_ctx(_session()))
    assert isinstance(result, dict)
    assert "error" not in result
    assert result.get("section_count", 0) > 0
    assert isinstance(result.get("sections"), list) and result["sections"]


def test_plan_arrangement_default_arc_starts_intro_ends_outro():
    result = plan_arrangement(_ctx(_session()))
    sections = result["sections"]
    first = sections[0]
    last = sections[-1]
    # SectionPlan.to_dict exposes the type; accept either 'section_type' or 'type'
    first_type = str(first.get("section_type") or first.get("type") or "").lower()
    last_type = str(last.get("section_type") or last.get("type") or "").lower()
    assert first_type == "intro"
    assert last_type == "outro"


def test_plan_arrangement_accepts_free_text_style():
    """Any style string is accepted (no VALID_STYLES gate) and echoed back."""
    result = plan_arrangement(_ctx(_session()), style="some-weird-genre")
    assert "error" not in result
    assert result.get("style") == "some-weird-genre"


def test_plan_arrangement_honors_explicit_sections():
    """Caller-supplied form is used instead of the default arc."""
    sections = [
        {"type": "intro", "energy": 0.2, "density": 0.2, "bars": 4},
        {"type": "drop", "energy": 1.0, "density": 1.0, "bars": 8},
        {"type": "outro", "energy": 0.2, "density": 0.1, "bars": 4},
    ]
    result = plan_arrangement(_ctx(_session()), sections=sections)
    assert "error" not in result
    assert result["section_count"] == 3


def test_plan_arrangement_tuple_sections_form():
    """The list/tuple entry form also works."""
    sections = [
        ["intro", 0.2, 0.2, 4],
        ["build", 0.6, 0.6, 8],
        ["outro", 0.2, 0.1, 4],
    ]
    result = plan_arrangement(_ctx(_session()), sections=sections)
    assert "error" not in result
    assert result["section_count"] == 3


def test_plan_arrangement_zero_bars_does_not_crash():
    """LIVE6-1: bars=0 in supplied sections must not raise ZeroDivisionError —
    bars are clamped to >=1 so template_bars is never 0."""
    result = plan_arrangement(_ctx(_session()), sections=[{"type": "intro", "bars": 0}])
    assert isinstance(result, dict)
    assert "error" not in result
    assert result.get("section_count", 0) >= 1


def test_plan_arrangement_malformed_sections_returns_structured_error():
    """LIVE6-2: a malformed sections entry (wrong arity / non-numeric) returns a
    structured {error, code} dict instead of escaping as a raw exception."""
    # wrong arity (3-tuple instead of 4)
    r1 = plan_arrangement(_ctx(_session()), sections=[["intro", 0.2, 0.2]])
    assert r1.get("code") == "INVALID_PARAM"
    assert "error" in r1
    # non-numeric energy
    r2 = plan_arrangement(_ctx(_session()), sections=[{"type": "intro", "energy": "loud"}])
    assert r2.get("code") == "INVALID_PARAM"

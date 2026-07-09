"""Regression test for BUG-FULL-MODE-15, rescoped to ComposerEngine.compose()
(mcp_server/composer/full/engine.py), reachable via commit_experiment ->
escalate_composer_branch -> ComposerEngine.compose().

apply_full_plan_v2's variant-slot flow already eliminated this bug for the
agent-designed path (see tests/composer/full/test_no_index_cascade.py), but
the deterministic ComposerEngine.compose() still assigned `track_index =
layer_idx` — a layer's position in the ORIGINAL (pre-drop) layer list —
regardless of whether earlier layers actually got a track created for them.
When a middle layer dropped as unresolved, later layers kept their stale
original index and the plan's create_midi_track(index=N) steps would fail
against the real Remote Script (which only has as many tracks as were
actually created).
"""

from __future__ import annotations

import asyncio

import pytest

from mcp_server.composer.full.engine import ComposerEngine
from mcp_server.composer.full.layer_planner import LayerSpec
from mcp_server.composer.prompt_parser import CompositionIntent


def _make_layers():
    # 4 layers; layer index 1 ("bass") will fail resolution below.
    return [
        LayerSpec(role="drums", search_query="drums", sections=["main"]),
        LayerSpec(role="bass", search_query="bass", sections=["main"]),
        LayerSpec(role="pad", search_query="pad", sections=["main"]),
        LayerSpec(role="texture", search_query="texture", sections=["main"]),
    ]


@pytest.mark.asyncio
async def test_track_indices_are_contiguous_after_unresolved_drop(monkeypatch):
    import mcp_server.composer.full.engine as engine_module

    layers = _make_layers()
    sections = [{"name": "main", "start_bar": 0, "bars": 4}]

    monkeypatch.setattr(engine_module, "plan_layers", lambda intent: layers)
    monkeypatch.setattr(engine_module, "plan_sections", lambda intent: sections)

    async def fake_resolve(layer, **kwargs):
        if layer.role == "bass":
            return (None, "unresolved")  # drops this layer
        return (f"/fake/{layer.role}.wav", "filesystem")

    monkeypatch.setattr(engine_module, "resolve_sample_for_layer", fake_resolve)

    engine = ComposerEngine()
    result = await engine.compose(intent=CompositionIntent(genre="techno", tempo=128))

    # bass dropped; drums/pad/texture survive
    assert "bass" in " ".join(result.warnings).lower()
    assert set(result.resolved_samples.keys()) == {"drums", "pad", "texture"}

    track_steps = [
        s for s in result.plan
        if s["tool"] == "create_midi_track"
    ]
    # 3 surviving layers -> 3 create_midi_track steps
    assert len(track_steps) == 3
    indices = [s["params"]["index"] for s in track_steps]
    # Contiguous, starting at 0 — NOT the stale original layer_idx
    # positions (which would have been 0, 2, 3).
    assert indices == [0, 1, 2], (
        f"Expected contiguous track indices [0, 1, 2] for the 3 surviving "
        f"layers, got {indices!r} (BUG-FULL-MODE-15 regression: stale "
        f"non-contiguous indices from dropped layers)"
    )

    # Every OTHER step referencing track_index for a given role must use
    # the SAME renumbered index as that role's create_midi_track step —
    # e.g. set_track_name for "pad" must reference index 1, not 2.
    role_to_index = {}
    for step in track_steps:
        role_to_index[step["role"]] = step["params"]["index"]
    for step in result.plan:
        role = step.get("role")
        if role in role_to_index and "track_index" in step.get("params", {}):
            assert step["params"]["track_index"] == role_to_index[role], (
                f"Step {step['tool']!r} for role {role!r} references "
                f"track_index={step['params']['track_index']!r} but that "
                f"role's track was created at index {role_to_index[role]!r}"
            )


@pytest.mark.asyncio
async def test_no_drops_still_yields_sequential_indices(monkeypatch):
    """Sanity check: with nothing dropped, behavior is unchanged (0,1,2,3)."""
    import mcp_server.composer.full.engine as engine_module

    layers = _make_layers()
    sections = [{"name": "main", "start_bar": 0, "bars": 4}]

    monkeypatch.setattr(engine_module, "plan_layers", lambda intent: layers)
    monkeypatch.setattr(engine_module, "plan_sections", lambda intent: sections)

    async def fake_resolve(layer, **kwargs):
        return (f"/fake/{layer.role}.wav", "filesystem")

    monkeypatch.setattr(engine_module, "resolve_sample_for_layer", fake_resolve)

    engine = ComposerEngine()
    result = await engine.compose(intent=CompositionIntent(genre="techno", tempo=128))

    track_steps = [s for s in result.plan if s["tool"] == "create_midi_track"]
    indices = [s["params"]["index"] for s in track_steps]
    assert indices == [0, 1, 2, 3]


@pytest.mark.asyncio
async def test_engine_no_longer_emits_fake_per_section_arrangement_clips(monkeypatch):
    """BUG-FULL-MODE-18 (rescoped): ComposerEngine.compose() has no
    per-section creative content — it used to tile the SAME 1-bar/single-
    note source clip across every section via create_arrangement_clip,
    which looks like a real arrangement but is actually pattern
    multiplication. The engine now emits only the Session-View scaffold
    (source clip + trigger note) and is honest about it via a warning,
    instead of faking arrangement placement.
    """
    import mcp_server.composer.full.engine as engine_module

    layers = [LayerSpec(role="drums", search_query="drums", sections=["intro", "drop"])]
    sections = [
        {"name": "intro", "start_bar": 0, "bars": 4},
        {"name": "drop", "start_bar": 4, "bars": 8},
    ]

    monkeypatch.setattr(engine_module, "plan_layers", lambda intent: layers)
    monkeypatch.setattr(engine_module, "plan_sections", lambda intent: sections)

    async def fake_resolve(layer, **kwargs):
        return (f"/fake/{layer.role}.wav", "filesystem")

    monkeypatch.setattr(engine_module, "resolve_sample_for_layer", fake_resolve)

    engine = ComposerEngine()
    result = await engine.compose(intent=CompositionIntent(genre="techno", tempo=128))

    tools_used = [s["tool"] for s in result.plan]
    assert "create_arrangement_clip" not in tools_used, (
        "Engine should no longer fake per-section arrangement placement"
    )
    # The Session-View scaffold (source clip + trigger note) is still there.
    assert "create_clip" in tools_used
    assert "add_notes" in tools_used
    # Callers get an explicit heads-up that Arrangement wasn't touched.
    assert any("arrangement" in w.lower() for w in result.warnings)

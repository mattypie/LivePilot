"""P1 (composer-broken): plan_sections must DEGRADE to a single full-length
section when no template is supplied (v1.24 removed SECTION_TEMPLATES) instead
of raising. Otherwise augment_with_samples / get_composition_plan crash and
propose_composer_branches silently returns zero branches (the ValueError is
swallowed by the per-strategy try/except).
"""

from mcp_server.composer.prompt_parser import CompositionIntent
from mcp_server.composer.full.layer_planner import plan_sections, plan_layers
from mcp_server.composer.branch_producer import propose_composer_branches


def test_plan_sections_degrades_without_template():
    intent = CompositionIntent(genre="techno", duration_bars=64)
    sections = plan_sections(intent)  # must NOT raise
    assert len(sections) == 1
    assert sections[0]["name"] == "Full"
    assert sections[0]["bars"] >= 4
    assert sections[0]["start_bar"] == 0


def test_plan_sections_still_honors_explicit_template():
    intent = CompositionIntent(genre="techno", duration_bars=64)
    template = [
        {"name": "Intro", "bars": 16, "layers": ["drums"]},
        {"name": "Main", "bars": 48, "layers": ["drums", "bass"]},
    ]
    sections = plan_sections(intent, section_template=template)
    assert [s["name"] for s in sections] == ["Intro", "Main"]


def test_plan_layers_works_without_template():
    intent = CompositionIntent(genre="techno", duration_bars=64)
    layers = plan_layers(intent)  # must NOT raise
    assert len(layers) >= 1
    for layer in layers:
        assert "Full" in layer.sections


def test_propose_composer_branches_returns_nonempty():
    pairs = propose_composer_branches("make a deep techno track", count=2)
    assert len(pairs) >= 1
    for _seed, plan in pairs:
        assert plan["step_count"] >= 1
        assert plan["steps"]

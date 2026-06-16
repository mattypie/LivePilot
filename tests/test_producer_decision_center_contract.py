"""v1.27 Producer Decision Center contract tests.

These tests keep the user-facing creative discipline from regressing back to
generic instrument defaults or effects-only "sound design".
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8").lower()


def test_creative_director_names_producer_decision_center():
    body = _read("livepilot/skills/livepilot-creative-director/SKILL.md")
    assert "producer decision center" in body
    assert "search_browser(path=\"sounds\"" in body
    assert "atlas_search" in body
    assert "atlas_device_info" in body


def test_creative_director_blocks_generic_default_instruments():
    body = _read("livepilot/skills/livepilot-creative-director/SKILL.md")
    assert "analog/poli/drift" in body
    assert "unless the user explicitly asked" in body
    assert "generic ai synth" in body


def test_creative_director_requires_source_level_sound_design_before_effects():
    body = _read("livepilot/skills/livepilot-creative-director/SKILL.md")
    assert "instrument/source-level" in body
    assert "before effects-only polish" in body
    assert "envelopes" in body
    assert "lfo routing" in body
    assert "sample start" in body


def test_capability_modes_doc_includes_link_and_stem_probe_domains():
    body = _read("livepilot/skills/livepilot-evaluation/references/capability-modes.md")
    assert "link_audio" in body
    assert "stem_workflow" in body
    assert "manual_only" in body
    assert "probe_link_audio" in body
    assert "probe_stem_workflow" in body

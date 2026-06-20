"""P1 response-cap regressions: extract_piano_roll budget/resolution guard,
extension_atlas_search body omission, corpus_emit_synthesis_briefs inline cap.

Each test FAILS before the fix and PASSES after.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _create_test_midi(path: str, notes=None, tempo=120.0):
    from midiutil import MIDIFile
    midi = MIDIFile(1)
    midi.addTempo(0, 0, tempo)
    if notes is None:
        notes = [(60 + i, i * 0.5, 0.5, 100) for i in range(8)]
    for pitch, start, dur, vel in notes:
        midi.addNote(0, 0, pitch, start, dur, vel)
    with open(path, "wb") as f:
        midi.writeFile(f)


# ── extract_piano_roll ────────────────────────────────────────────────────


def test_piano_roll_rejects_nonpositive_resolution():
    """resolution<=0 must return a structured error, not raise ZeroDivisionError."""
    from mcp_server.tools.midi_io import extract_piano_roll

    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        path = f.name
    try:
        _create_test_midi(path)
        result = extract_piano_roll(ctx=None, file_path=path, resolution=0.0)
        assert "error" in result
        assert result.get("code") == "INVALID_PARAM"
        assert "piano_roll" not in result
    finally:
        os.unlink(path)


def test_piano_roll_caps_oversized_matrix():
    """A long MIDI at fine resolution must NOT return an unbounded matrix."""
    from mcp_server.tools.midi_io import extract_piano_roll, _PIANO_ROLL_CELL_BUDGET

    # ~400 beats of sustained notes spanning a wide pitch range; at a 32nd-note
    # resolution this is tens of thousands of cells -> over budget.
    notes = []
    for i in range(40):
        pitch = 36 + i  # wide pitch span
        notes.append((pitch, float(i), 400.0, 100))
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        path = f.name
    try:
        _create_test_midi(path, notes=notes, tempo=120.0)
        result = extract_piano_roll(ctx=None, file_path=path, resolution=0.0625)
        assert "error" in result, "oversized roll should be rejected, not materialized"
        assert result.get("code") == "INVALID_PARAM"
        assert "piano_roll" not in result
        # The reported dims must actually exceed the budget.
        assert result["pitch_range"] * result["time_steps"] > _PIANO_ROLL_CELL_BUDGET
    finally:
        os.unlink(path)


def test_piano_roll_small_input_still_returns_matrix():
    """Small inputs are unaffected and still return the full matrix."""
    from mcp_server.tools.midi_io import extract_piano_roll

    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        path = f.name
    try:
        _create_test_midi(path)  # 8 short notes
        result = extract_piano_roll(ctx=None, file_path=path, resolution=0.5)
        assert "piano_roll" in result
        assert result["pitch_min"] == 60
        assert result["pitch_max"] == 67
    finally:
        os.unlink(path)


# ── extension_atlas_search ────────────────────────────────────────────────


def _seed_big_overlay(monkeypatch, tmp_path: Path):
    fake_home = tmp_path / "fake_home"
    overlay_root = fake_home / ".livepilot" / "atlas-overlays" / "elektron"
    overlay_root.mkdir(parents=True)
    big_arch = "x" * 5000  # a large body field
    (overlay_root / "fixture.yaml").write_text(
        "- entity_id: sophie_ponyboy_kick\n"
        "  entity_type: signature_chain\n"
        "  name: SOPHIE Ponyboy kick\n"
        "  description: 3-track recipe\n"
        "  tags: [kick, sophie]\n"
        "  artists: [sophie_xeon]\n"
        f"  architecture: \"{big_arch}\"\n"
    )
    monkeypatch.setattr(Path, "home", lambda: fake_home)
    from mcp_server.atlas.overlays import load_overlays
    load_overlays()


def test_search_omits_full_body(monkeypatch, tmp_path):
    """Search results must NOT inline the heavy YAML body."""
    _seed_big_overlay(monkeypatch, tmp_path)
    from mcp_server.atlas.tools import extension_atlas_search

    result = extension_atlas_search(ctx=None, query="sophie")
    assert result["count"] == 1
    entry = result["results"][0]
    assert entry["entity_id"] == "sophie_ponyboy_kick"
    assert "body" not in entry, "search result must not carry the full body"
    # the 5KB architecture blob must not have leaked into the serialized result
    assert "x" * 5000 not in json.dumps(result)


def test_get_still_returns_full_body(monkeypatch, tmp_path):
    """The get path must STILL return the full body (contract preserved)."""
    _seed_big_overlay(monkeypatch, tmp_path)
    from mcp_server.atlas.tools import extension_atlas_get

    entry = extension_atlas_get(ctx=None, namespace="elektron",
                                entity_id="sophie_ponyboy_kick")
    assert entry["entity_id"] == "sophie_ponyboy_kick"
    assert entry["body"]["architecture"] == "x" * 5000


# ── corpus_emit_synthesis_briefs ──────────────────────────────────────────


def test_emit_briefs_caps_inline(monkeypatch, tmp_path):
    """With more plugins than inline_limit, full briefs are capped and the
    overflow comes back as lightweight stubs."""
    import mcp_server.user_corpus.tools as uct

    out_root = tmp_path / "out"
    plugins_dir = out_root / "plugins"
    plugins_dir.mkdir(parents=True)

    plugins = []
    for i in range(12):
        plugins.append({
            "plugin_id": f"pid_{i}",
            "name": f"Plugin {i}",
            "vendor": "Acme",
            "format": "VST3",
            "version": "1.0",
            "bundle_path": f"/fake/Plugin{i}.vst3",
            "unique_id": f"UID{i}",
        })
    (plugins_dir / "_inventory.json").write_text(json.dumps({"plugins": plugins}))

    monkeypatch.setattr(uct, "DEFAULT_OUTPUT_ROOT", out_root)

    result = uct.corpus_emit_synthesis_briefs(ctx=None, inline_limit=5)
    assert result["total"] == 12
    assert result["inline_count"] == 5
    assert len(result["briefs"]) == 5
    assert len(result["deferred"]) == 7
    # deferred entries are stubs: no heavy 'brief' payload
    for stub in result["deferred"]:
        assert "brief" not in stub
        assert "plugin_id" in stub and "output_path" in stub
    # inline entries DO carry the full brief
    assert "brief" in result["briefs"][0]


def test_emit_briefs_explicit_ids_unbounded_by_default_cap(monkeypatch, tmp_path):
    """Selecting fewer plugins than the cap returns them all inline, no deferral."""
    import mcp_server.user_corpus.tools as uct

    out_root = tmp_path / "out2"
    plugins_dir = out_root / "plugins"
    plugins_dir.mkdir(parents=True)
    plugins = [{
        "plugin_id": f"pid_{i}", "name": f"P{i}", "vendor": "Acme",
        "format": "VST3", "version": "1.0",
        "bundle_path": f"/fake/P{i}.vst3", "unique_id": f"UID{i}",
    } for i in range(12)]
    (plugins_dir / "_inventory.json").write_text(json.dumps({"plugins": plugins}))
    monkeypatch.setattr(uct, "DEFAULT_OUTPUT_ROOT", out_root)

    result = uct.corpus_emit_synthesis_briefs(
        ctx=None, plugin_ids=["pid_0", "pid_3", "pid_7"], inline_limit=5)
    assert result["inline_count"] == 3
    assert result["deferred"] == []
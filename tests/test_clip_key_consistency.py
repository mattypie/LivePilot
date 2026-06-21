"""Regression tests for check_clip_key_consistency (.fn AttributeError bug).

Under FastMCP 3.3.1, @mcp.tool() returns the plain async function with no
`.fn` attribute. The tool used to call `get_clip_file_path.fn(...)` /
`get_detected_key.fn(...)`, which raised AttributeError that was swallowed
into status="unknown" — permanently breaking the tool.
"""
from __future__ import annotations

import asyncio

import mcp_server.tools.analyzer as analyzer_mod
from mcp_server.tools.clips import check_clip_key_consistency


class _DummyCtx:
    """Stand-in for FastMCP Context; the patched callees ignore it."""
    pass


def _run(coro):
    return asyncio.run(coro)


def test_check_clip_key_consistency_reaches_real_comparison(monkeypatch):
    """With the bridge/analyzer mocked, the tool must compute a real mismatch,
    not fall back to status='unknown' via a swallowed AttributeError."""

    async def fake_get_path(ctx, track_index, clip_index):
        # Splice-style filename encoding D#min (semi=3, minor).
        return {"path": "/samples/AU_THF2_128_vocal_loop_D#min.wav"}

    async def fake_get_key(ctx):
        # Session detected as D minor (semi=2).
        return {"key": "D", "scale": "minor"}

    monkeypatch.setattr(analyzer_mod, "get_clip_file_path", fake_get_path)
    monkeypatch.setattr(analyzer_mod, "get_detected_key", fake_get_key)

    result = _run(check_clip_key_consistency(_DummyCtx(), 6, 0))

    assert isinstance(result, dict)
    # Before the fix this was "unknown" with reason starting
    # "Could not resolve clip file path".
    assert result["status"] == "mismatch", result
    # D#min (3) vs D min (2): clip should shift DOWN 1 semitone.
    assert result["semitone_delta"] == -1, result
    assert result["recommended_fix"]["tool"] == "set_clip_pitch"
    assert result["recommended_fix"]["args"]["coarse"] == -1


def test_check_clip_key_consistency_no_swallowed_attributeerror(monkeypatch):
    """The 'unknown' fallback must come from genuine missing data, never
    from an AttributeError on a non-existent .fn accessor."""

    async def fake_get_path(ctx, track_index, clip_index):
        # MIDI clip / no audio path → legitimate unknown.
        return {"error": "No file path available (MIDI clip?)."}

    monkeypatch.setattr(analyzer_mod, "get_clip_file_path", fake_get_path)

    result = _run(check_clip_key_consistency(_DummyCtx(), 0, 0))

    assert result["status"] == "unknown"
    # Must NOT be the AttributeError-driven fallback.
    assert "has no attribute" not in result.get("reason", "")
    assert not result.get("reason", "").startswith("Could not resolve clip file path")
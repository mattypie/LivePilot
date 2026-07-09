"""P3-47 — atlas scan truncation observability.

DEEP_REVIEW_2026-06-21 P3-47 / DEEP_REVIEW_2026-07-09 P1-11: scan_browser_deep
defaulted max_per_category=1000 and reset its iteration safety counter per
category, so a fresh scan_full_library silently truncated large
alphabetically-ordered browser categories (drum_kits: 0 kicks / 2 hats / 9
snares; sounds cut off inside "Brass"). This suite covers the MCP-side half
of the fix:

1. scan_full_library's own default raised to match the remote script (25000)
   and passed through explicitly to scan_browser_deep.
2. Truncation flags/counts from scan_browser_deep are persisted into
   device_atlas.json's `stats` (category_truncated remapped to the atlas
   device-category vocabulary, e.g. "drums" -> "drum_kits").
3. Backward compatibility: a pre-update remote script that only echoes
   `counts` (no `category_truncated` map) still gets a correct fallback via
   the count>=cap heuristic.
4. AtlasManager surfaces the persisted truncation flags via
   truncated_categories()/truncation_warning(), and atlas_search /
   atlas_suggest attach a "warning" field when a query touches a category
   the last scan had to truncate.

The remote-script side (global iteration counter, raised default, per-
category counts/category_truncated in scan_browser_deep's own return value)
is covered separately in tests/test_browser_scan_deep.py.
"""

from __future__ import annotations

import importlib
import inspect
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from mcp_server.atlas import AtlasManager
from mcp_server.atlas import tools as atlas_tools


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_atlas(tmp_path: Path, devices: list, category_truncated: dict | None = None) -> str:
    """Write a minimal device_atlas.json and return its path."""
    path = tmp_path / "device_atlas.json"
    stats: dict = {"total_devices": len(devices)}
    if category_truncated is not None:
        stats["category_truncated"] = category_truncated
    path.write_text(json.dumps({
        "version": "2.0.0",
        "devices": devices,
        "stats": stats,
    }))
    return str(path)


# ── scan_full_library: raised default + explicit pass-through ──────────


def test_scan_full_library_default_max_per_category_is_25000():
    """The MCP-side default must match the remote script's raised default
    (25000) — otherwise a caller who omits max_per_category still gets the
    old truncation behavior even after the remote script is updated."""
    sig = inspect.signature(atlas_tools.scan_full_library)
    assert sig.parameters["max_per_category"].default == 25000


def test_scan_full_library_passes_max_per_category_explicitly(tmp_path, monkeypatch):
    """scan_full_library must forward whatever max_per_category it was
    given (default or explicit) to scan_browser_deep, so a caller running
    an OLDER remote script (still defaulting to 1000) gets the caller's
    value rather than the remote's stale default."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    for name in list(sys.modules):
        if name == "mcp_server.atlas" or name.startswith("mcp_server.atlas."):
            del sys.modules[name]
    sys.path.insert(0, str(REPO_ROOT))
    try:
        tools_mod = importlib.import_module("mcp_server.atlas.tools")
    finally:
        sys.path.remove(str(REPO_ROOT))

    captured: dict = {}

    class _FakeAbleton:
        def send_command(self, cmd, payload=None):
            if cmd == "scan_browser_deep":
                captured["payload"] = payload
                return {"categories": {}, "counts": {}, "category_truncated": {}}
            if cmd == "get_session_info":
                return {"live_version": "12.4.0"}
            return {}

    monkeypatch.setattr(tools_mod, "_get_ableton", lambda ctx: _FakeAbleton())

    tools_mod.scan_full_library(ctx=None, force=True)
    assert captured["payload"] == {"max_per_category": 25000}

    tools_mod.scan_full_library(ctx=None, force=True, max_per_category=7)
    assert captured["payload"] == {"max_per_category": 7}


# ── Truncation flags persisted into device_atlas.json ───────────────────


def test_scan_full_library_persists_category_truncated_map(tmp_path, monkeypatch):
    """The explicit category_truncated map from an updated remote script
    must land in stats.category_truncated, remapped through the scanner's
    raw->atlas category vocabulary (raw "drums" -> atlas "drum_kits")."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    for name in list(sys.modules):
        if name == "mcp_server.atlas" or name.startswith("mcp_server.atlas."):
            del sys.modules[name]
    sys.path.insert(0, str(REPO_ROOT))
    try:
        atlas_mod = importlib.import_module("mcp_server.atlas")
        tools_mod = importlib.import_module("mcp_server.atlas.tools")
    finally:
        sys.path.remove(str(REPO_ROOT))

    class _FakeAbleton:
        def send_command(self, cmd, payload=None):
            if cmd == "scan_browser_deep":
                return {
                    "categories": {
                        "drums": [
                            {"name": "Crash", "uri": "u:1", "is_loadable": True},
                        ],
                        "instruments": [
                            {"name": "Operator", "uri": "u:2", "is_loadable": True},
                        ],
                    },
                    "counts": {"drums": 1, "instruments": 1},
                    "category_truncated": {"drums": True, "instruments": False},
                }
            if cmd == "get_session_info":
                return {"live_version": "12.4.0"}
            return {}

    monkeypatch.setattr(tools_mod, "_get_ableton", lambda ctx: _FakeAbleton())
    result = tools_mod.scan_full_library(ctx=None, force=True)

    assert result["status"] == "scanned"
    written = json.loads(atlas_mod.USER_ATLAS_PATH.read_text())
    assert written["truncated_categories"] == ["drums"]
    assert written["stats"]["category_truncated"] == {"drum_kits": True}
    assert written["stats"]["category_counts"] == {"drums": 1, "instruments": 1}


def test_scan_full_library_falls_back_to_count_heuristic_for_old_remote_scripts(
    tmp_path, monkeypatch
):
    """A pre-update remote script only echoes `counts` (no
    `category_truncated`); scan_full_library must still detect truncation
    via the count>=cap heuristic and remap it into the atlas vocabulary."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    for name in list(sys.modules):
        if name == "mcp_server.atlas" or name.startswith("mcp_server.atlas."):
            del sys.modules[name]
    sys.path.insert(0, str(REPO_ROOT))
    try:
        atlas_mod = importlib.import_module("mcp_server.atlas")
        tools_mod = importlib.import_module("mcp_server.atlas.tools")
    finally:
        sys.path.remove(str(REPO_ROOT))

    class _FakeAbleton:
        def send_command(self, cmd, payload=None):
            if cmd == "scan_browser_deep":
                return {
                    "categories": {
                        "drums": [
                            {"name": "d%d" % i, "uri": "u%d" % i, "is_loadable": True}
                            for i in range(3)
                        ],
                    },
                    "counts": {"drums": 3},
                    # no category_truncated key — old remote script shape
                }
            if cmd == "get_session_info":
                return {"live_version": "12.4.0"}
            return {}

    monkeypatch.setattr(tools_mod, "_get_ableton", lambda ctx: _FakeAbleton())
    result = tools_mod.scan_full_library(ctx=None, force=True, max_per_category=3)

    assert result["status"] == "scanned"
    written = json.loads(atlas_mod.USER_ATLAS_PATH.read_text())
    assert written["truncated_categories"] == ["drums"]
    assert written["stats"]["category_truncated"] == {"drum_kits": True}


def test_scan_full_library_no_truncation_flags_when_nothing_capped(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    for name in list(sys.modules):
        if name == "mcp_server.atlas" or name.startswith("mcp_server.atlas."):
            del sys.modules[name]
    sys.path.insert(0, str(REPO_ROOT))
    try:
        atlas_mod = importlib.import_module("mcp_server.atlas")
        tools_mod = importlib.import_module("mcp_server.atlas.tools")
    finally:
        sys.path.remove(str(REPO_ROOT))

    class _FakeAbleton:
        def send_command(self, cmd, payload=None):
            if cmd == "scan_browser_deep":
                return {
                    "categories": {"instruments": [
                        {"name": "Operator", "uri": "u:1", "is_loadable": True},
                    ]},
                    "counts": {"instruments": 1},
                    "category_truncated": {"instruments": False},
                }
            if cmd == "get_session_info":
                return {"live_version": "12.4.0"}
            return {}

    monkeypatch.setattr(tools_mod, "_get_ableton", lambda ctx: _FakeAbleton())
    tools_mod.scan_full_library(ctx=None, force=True)

    written = json.loads(atlas_mod.USER_ATLAS_PATH.read_text())
    assert written["truncated_categories"] == []
    assert written["stats"]["category_truncated"] == {}


# ── AtlasManager: truncated_categories() / truncation_warning() ─────────


def test_truncated_categories_reads_persisted_stats(tmp_path):
    path = _write_atlas(
        tmp_path, devices=[],
        category_truncated={"drum_kits": True, "sounds": False},
    )
    mgr = AtlasManager(path)
    assert mgr.truncated_categories() == ["drum_kits"]


def test_truncation_warning_none_without_truncation_stats(tmp_path):
    path = _write_atlas(tmp_path, devices=[])
    mgr = AtlasManager(path)
    assert mgr.truncated_categories() == []
    assert mgr.truncation_warning("all") is None
    assert mgr.truncation_warning("drum_kits") is None


def test_truncation_warning_matches_exact_category(tmp_path):
    path = _write_atlas(tmp_path, devices=[], category_truncated={"drum_kits": True})
    mgr = AtlasManager(path)
    assert mgr.truncation_warning("drum_kits") is not None
    assert mgr.truncation_warning("sounds") is None


def test_truncation_warning_all_fires_on_any_truncated_category(tmp_path):
    path = _write_atlas(tmp_path, devices=[], category_truncated={"sounds": True})
    mgr = AtlasManager(path)
    assert mgr.truncation_warning("all") is not None


def test_truncation_warning_respects_category_aliases(tmp_path):
    """"instrument" (singular) must warn when "instruments" is flagged —
    matches the same alias table search() uses for filtering."""
    path = _write_atlas(tmp_path, devices=[], category_truncated={"instruments": True})
    mgr = AtlasManager(path)
    assert mgr.truncation_warning("instrument") is not None
    assert mgr.truncation_warning("instruments") is not None
    assert mgr.truncation_warning("effects") is None


# ── atlas_search / atlas_suggest: "warning" field wiring ─────────────────


OPERATOR_DEVICE = {
    "id": "operator",
    "name": "Operator",
    "uri": "query:Synths#Operator",
    "category": "instruments",
    "character_tags": ["fm", "digital"],
    "use_cases": ["bass"],
}


def test_atlas_search_appends_warning_for_truncated_category(tmp_path):
    path = _write_atlas(
        tmp_path, devices=[OPERATOR_DEVICE],
        category_truncated={"instruments": True},
    )
    mgr = AtlasManager(path)
    with patch.object(atlas_tools, "_get_atlas", return_value=mgr):
        result = atlas_tools.atlas_search(ctx=None, query="Operator", category="instruments")
    assert "warning" in result
    assert "instruments" in result["warning"]


def test_atlas_search_no_warning_for_untruncated_category(tmp_path):
    path = _write_atlas(
        tmp_path, devices=[OPERATOR_DEVICE],
        category_truncated={"drum_kits": True},
    )
    mgr = AtlasManager(path)
    with patch.object(atlas_tools, "_get_atlas", return_value=mgr):
        result = atlas_tools.atlas_search(ctx=None, query="Operator", category="instruments")
    assert "warning" not in result


def test_atlas_search_no_warning_when_atlas_has_no_truncation_stats(tmp_path):
    path = _write_atlas(tmp_path, devices=[OPERATOR_DEVICE])
    mgr = AtlasManager(path)
    with patch.object(atlas_tools, "_get_atlas", return_value=mgr):
        result = atlas_tools.atlas_search(ctx=None, query="Operator")
    assert "warning" not in result


def test_atlas_suggest_appends_warning_when_any_category_truncated(tmp_path):
    path = _write_atlas(
        tmp_path, devices=[OPERATOR_DEVICE],
        category_truncated={"drum_kits": True},
    )
    mgr = AtlasManager(path)
    with patch.object(atlas_tools, "_get_atlas", return_value=mgr):
        result = atlas_tools.atlas_suggest(ctx=None, intent="Operator bass")
    assert "warning" in result


def test_atlas_suggest_no_warning_when_atlas_has_no_truncation_stats(tmp_path):
    path = _write_atlas(tmp_path, devices=[OPERATOR_DEVICE])
    mgr = AtlasManager(path)
    with patch.object(atlas_tools, "_get_atlas", return_value=mgr):
        result = atlas_tools.atlas_suggest(ctx=None, intent="Operator bass")
    assert "warning" not in result

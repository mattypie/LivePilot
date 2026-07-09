"""Handler-level tests for remote_script.LivePilot.browser covering the
commands NOT already exercised by tests/test_browser_scan_deep.py (which
only drives scan_browser_deep). This file covers: get_browser_tree,
get_browser_items, search_browser, load_browser_item (URI hit, name
fallback, FileId deep-scan budget, not-found), and get_device_presets.

Uses the same fake-LOM importlib pattern as
tests/test_remote_script_contracts.py's `_load_remote_devices` — browser.py
does `import Live` at module scope, so a bare stub module is injected
before loading.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = ROOT / "remote_script" / "LivePilot"


def _load_remote_browser():
    for name in [
        "remote_script.LivePilot.browser",
        "remote_script.LivePilot.router",
        "remote_script.LivePilot.utils",
        "remote_script.LivePilot",
        "remote_script",
    ]:
        sys.modules.pop(name, None)

    sys.modules.setdefault("Live", types.ModuleType("Live"))

    remote_pkg = types.ModuleType("remote_script")
    remote_pkg.__path__ = [str(ROOT / "remote_script")]
    sys.modules["remote_script"] = remote_pkg

    live_pkg = types.ModuleType("remote_script.LivePilot")
    live_pkg.__path__ = [str(REMOTE_ROOT)]
    sys.modules["remote_script.LivePilot"] = live_pkg

    def _load(name: str, path: Path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    _load("remote_script.LivePilot.utils", REMOTE_ROOT / "utils.py")
    router = _load("remote_script.LivePilot.router", REMOTE_ROOT / "router.py")
    browser = _load("remote_script.LivePilot.browser", REMOTE_ROOT / "browser.py")
    return router, browser


class _FakeItem:
    def __init__(self, name, is_loadable=False, is_folder=False,
                 uri="uri:default", children=None):
        self.name = name
        self.is_loadable = is_loadable
        self.is_folder = is_folder
        self.uri = uri
        self.children = list(children) if children is not None else []


def _make_browser():
    """Build a minimal fake Browser with a couple of nested categories."""
    kick = _FakeItem("Kick 808", is_loadable=True, uri="query:Drums#Kick_808")
    snare = _FakeItem("Snare Tight", is_loadable=True, uri="query:Drums#Snare_Tight")
    drum_folder = _FakeItem("Kits", is_folder=True, children=[kick, snare])

    pad = _FakeItem("Pad Warm", is_loadable=True, uri="query:Sounds#Pad_Warm")
    sounds_root_children = [pad]

    class _Browser:
        instruments = _FakeItem("instruments", is_folder=True, children=[])
        audio_effects = _FakeItem("audio_effects", is_folder=True, children=[])
        midi_effects = _FakeItem("midi_effects", is_folder=True, children=[])
        sounds = _FakeItem("sounds", is_folder=True, children=sounds_root_children)
        drums = _FakeItem("drums", is_folder=True, children=[drum_folder])
        samples = _FakeItem("samples", is_folder=True, children=[])
        packs = _FakeItem("packs", is_folder=True, children=[])
        user_library = _FakeItem("user_library", is_folder=True, children=[])

        def load_item(self, item):
            self.loaded_item = item

    return _Browser()


class _FakeTrack:
    def __init__(self, name="Track 1", devices=None):
        self.name = name
        self.devices = list(devices) if devices is not None else []


class _FakeView:
    selected_track = None


class _FakeSong:
    def __init__(self, tracks):
        self.tracks = tracks
        self.master_track = None
        self.return_tracks = []
        self.view = _FakeView()


def _patch_get_browser(monkeypatch, browser_mod, browser):
    monkeypatch.setattr(browser_mod, "_get_browser", lambda: browser)


def test_get_browser_tree_returns_children_preview_and_counts(monkeypatch):
    _router, browser_mod = _load_remote_browser()
    browser = _make_browser()
    _patch_get_browser(monkeypatch, browser_mod, browser)

    result = browser_mod.get_browser_tree(None, {"category_type": "drums"})
    cats = {c["name"]: c for c in result["categories"]}
    assert "drums" in cats
    assert cats["drums"]["children_count"] == 1
    assert cats["drums"]["children_preview"] == ["Kits"]


def test_get_browser_tree_unknown_category_raises(monkeypatch):
    _router, browser_mod = _load_remote_browser()
    browser = _make_browser()
    _patch_get_browser(monkeypatch, browser_mod, browser)

    with pytest.raises(ValueError, match="Unknown category"):
        browser_mod.get_browser_tree(None, {"category_type": "nonexistent"})


def test_get_browser_items_lists_children_with_uri(monkeypatch):
    _router, browser_mod = _load_remote_browser()
    browser = _make_browser()
    _patch_get_browser(monkeypatch, browser_mod, browser)

    result = browser_mod.get_browser_items(None, {"path": "sounds"})
    assert result["path"] == "sounds"
    names = [i["name"] for i in result["items"]]
    assert names == ["Pad Warm"]
    assert result["items"][0]["uri"] == "query:Sounds#Pad_Warm"


def test_get_browser_items_unknown_path_segment_raises(monkeypatch):
    _router, browser_mod = _load_remote_browser()
    browser = _make_browser()
    _patch_get_browser(monkeypatch, browser_mod, browser)

    with pytest.raises(ValueError, match="not found"):
        browser_mod.get_browser_items(None, {"path": "drums/DoesNotExist"})


def test_search_browser_finds_nested_loadable_items_by_name_filter(monkeypatch):
    _router, browser_mod = _load_remote_browser()
    browser = _make_browser()
    _patch_get_browser(monkeypatch, browser_mod, browser)

    result = browser_mod.search_browser(None, {"path": "drums", "name_filter": "snare"})
    assert result["count"] == 1
    assert result["items"][0]["name"] == "Snare Tight"
    assert result["items"][0]["uri"] == "query:Drums#Snare_Tight"


def test_search_browser_loadable_only_filters_folders(monkeypatch):
    _router, browser_mod = _load_remote_browser()
    browser = _make_browser()
    _patch_get_browser(monkeypatch, browser_mod, browser)

    result = browser_mod.search_browser(None, {"path": "drums", "loadable_only": True})
    names = [i["name"] for i in result["items"]]
    # "Kits" folder itself is not loadable and must be excluded
    assert "Kits" not in names
    assert set(names) == {"Kick 808", "Snare Tight"}


def test_load_browser_item_matches_by_exact_uri(monkeypatch):
    _router, browser_mod = _load_remote_browser()
    browser = _make_browser()
    _patch_get_browser(monkeypatch, browser_mod, browser)
    track = _FakeTrack(devices=[])
    song = _FakeSong([track])

    # load_item appends a device to simulate Live's post-load state.
    def _load_item(item):
        track.devices.append(types.SimpleNamespace(name=item.name))
    browser.load_item = _load_item

    result = browser_mod.load_browser_item(
        song, {"track_index": 0, "uri": "query:Drums#Kick_808"}
    )
    assert result["loaded"] is True
    assert result["name"] == "Kick 808"
    assert result["device_count"] == 1
    assert result["device_index"] == 0  # appended at the end


def test_load_browser_item_falls_back_to_name_match(monkeypatch):
    """URI that doesn't exactly match any child.uri falls back to a
    name-extraction strategy (last path segment, stripped extensions)."""
    _router, browser_mod = _load_remote_browser()
    browser = _make_browser()
    _patch_get_browser(monkeypatch, browser_mod, browser)
    track = _FakeTrack(devices=[])
    song = _FakeSong([track])

    def _load_item(item):
        track.devices.append(types.SimpleNamespace(name=item.name))
    browser.load_item = _load_item

    # No child has this exact uri, but "Pad Warm" can be found via name match
    # once the fragment is extracted from the uri string.
    result = browser_mod.load_browser_item(
        song, {"track_index": 0, "uri": "somecategory:Pad Warm"}
    )
    assert result["loaded"] is True
    assert result["name"] == "Pad Warm"


def test_load_browser_item_not_found_raises_value_error(monkeypatch):
    _router, browser_mod = _load_remote_browser()
    browser = _make_browser()
    _patch_get_browser(monkeypatch, browser_mod, browser)
    track = _FakeTrack(devices=[])
    song = _FakeSong([track])

    with pytest.raises(ValueError, match="not found in browser"):
        browser_mod.load_browser_item(
            song, {"track_index": 0, "uri": "nope:TotallyMissingDevice"}
        )


def test_load_browser_item_fileid_uri_respects_deep_scan_budget(monkeypatch):
    """A FileId-style URI with no exact match must raise a clean STATE_ERROR-
    shaped ValueError pointing at search_browser(), not hang forever."""
    _router, browser_mod = _load_remote_browser()
    browser = _make_browser()
    _patch_get_browser(monkeypatch, browser_mod, browser)
    track = _FakeTrack(devices=[])
    song = _FakeSong([track])

    with pytest.raises(ValueError, match="deep-scan budget"):
        browser_mod.load_browser_item(
            song, {"track_index": 0, "uri": "query:Pad:FileId_999999"}
        )


def test_load_browser_item_invalid_track_index_raises_index_error(monkeypatch):
    _router, browser_mod = _load_remote_browser()
    browser = _make_browser()
    _patch_get_browser(monkeypatch, browser_mod, browser)
    song = _FakeSong([_FakeTrack()])

    with pytest.raises(IndexError):
        browser_mod.load_browser_item(
            song, {"track_index": 5, "uri": "query:Drums#Kick_808"}
        )


def test_get_device_presets_finds_named_device_and_collects_nested_presets(monkeypatch):
    _router, browser_mod = _load_remote_browser()

    preset_a = _FakeItem("Preset A", is_loadable=True, uri="uri:preset_a")
    nested_folder = _FakeItem("Default Presets", is_folder=True, is_loadable=True,
                               children=[preset_a])
    device_item = _FakeItem("Operator", is_folder=True, is_loadable=True,
                             children=[nested_folder])

    class _Browser:
        instruments = _FakeItem("instruments", is_folder=True, children=[device_item])
        audio_effects = _FakeItem("audio_effects", is_folder=True, children=[])
        midi_effects = _FakeItem("midi_effects", is_folder=True, children=[])

    _patch_get_browser(monkeypatch, browser_mod, _Browser())

    result = browser_mod.get_device_presets(None, {"device_name": "operator"})
    assert result["category"] == "instruments"
    assert [p["name"] for p in result["presets"]] == ["Preset A"]


def test_get_device_presets_device_not_found_returns_none_category(monkeypatch):
    _router, browser_mod = _load_remote_browser()

    class _Browser:
        instruments = _FakeItem("instruments", is_folder=True, children=[])
        audio_effects = _FakeItem("audio_effects", is_folder=True, children=[])
        midi_effects = _FakeItem("midi_effects", is_folder=True, children=[])

    _patch_get_browser(monkeypatch, browser_mod, _Browser())

    result = browser_mod.get_device_presets(None, {"device_name": "GhostSynth"})
    assert result["category"] is None
    assert result["presets"] == []

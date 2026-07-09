"""Regression tests for remote_script.LivePilot.browser's scan_browser_deep.

Covers the P3-47 fix (DEEP_REVIEW_2026-06-21 / DEEP_REVIEW_2026-07-09
P1-11): the 100-item-alphabetical-truncation bug where every top-level
browser category silently capped at the (formerly 1000-default)
`max_per_category`, and the iteration safety bound (`_SCAN_MAX_ITERATIONS`)
reset per category instead of being a true scan-wide budget.

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
    """Load remote_script.LivePilot.browser with Live stubbed.

    Mirrors test_remote_script_contracts.py's `_load_remote_devices` —
    browser.py does `import Live` at module top (for the Application
    object), so a bare ModuleType is injected before loading.
    """
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
    """Minimal stand-in for a Live.Browser.BrowserItem."""

    def __init__(self, name, is_loadable=False, is_folder=False,
                 uri="uri:default", children=None):
        self.name = name
        self.is_loadable = is_loadable
        self.is_folder = is_folder
        self.uri = uri
        self.children = list(children) if children is not None else []


def _flat_folder(name, count):
    """A folder containing `count` flat (non-folder) loadable children."""
    kids = [
        _FakeItem("%s_%d" % (name, i), is_loadable=True, uri="uri:%s_%d" % (name, i))
        for i in range(count)
    ]
    return _FakeItem(name, is_folder=True, children=kids)


class _FakeBrowser:
    """Stand-in for Live.Application.get_application().browser.

    Only the 8 base categories that `_get_categories` unconditionally
    reads are populated; the optional ones (plugins/max_for_live/clips/
    current_project) are read via getattr+AttributeError in production
    code and are simply absent here.
    """

    _BASE_CATEGORIES = (
        "instruments", "audio_effects", "midi_effects",
        "sounds", "drums", "samples", "packs", "user_library",
    )

    def __init__(self, **overrides):
        for cat in self._BASE_CATEGORIES:
            setattr(self, cat, overrides.get(cat, _FakeItem(cat, is_folder=True)))


@pytest.fixture
def loaded_browser():
    _router, browser = _load_remote_browser()
    return browser


# ── Default cap raised 1000 -> 25000 ────────────────────────────────────


def test_default_max_per_category_pins_at_25000(loaded_browser, monkeypatch):
    """A category with more than the OLD 1000 default (but <= 25000) must
    come back complete and unflagged when no max_per_category is passed."""
    browser = loaded_browser
    fake = _FakeBrowser(samples=_flat_folder("samples", 2000))
    monkeypatch.setattr(browser, "_get_browser", lambda: fake)

    result = browser.scan_browser_deep(None, {})

    assert len(result["categories"]["samples"]) == 2000
    assert result["counts"]["samples"] == 2000
    assert result["category_truncated"]["samples"] is False


def test_default_max_per_category_caps_exactly_at_25000(loaded_browser, monkeypatch):
    """Pin the exact new default: 25001 available items truncates to
    exactly 25000 and flags the category truncated."""
    browser = loaded_browser
    fake = _FakeBrowser(sounds=_flat_folder("sounds", 25001))
    monkeypatch.setattr(browser, "_get_browser", lambda: fake)

    result = browser.scan_browser_deep(None, {})

    assert len(result["categories"]["sounds"]) == 25000
    assert result["counts"]["sounds"] == 25000
    assert result["category_truncated"]["sounds"] is True


# ── Explicit max_per_category still respected ───────────────────────────


def test_explicit_max_per_category_still_caps(loaded_browser, monkeypatch):
    browser = loaded_browser
    fake = _FakeBrowser(drums=_flat_folder("drums", 5))
    monkeypatch.setattr(browser, "_get_browser", lambda: fake)

    result = browser.scan_browser_deep(None, {"max_per_category": 2})

    assert len(result["categories"]["drums"]) == 2
    assert result["counts"]["drums"] == 2
    assert result["category_truncated"]["drums"] is True


# ── P3-47: iteration counter is GLOBAL across categories ────────────────


def test_iteration_counter_is_shared_across_categories(loaded_browser, monkeypatch):
    """Regression pin for P3-47: before the fix, `_counter` defaulted to a
    fresh [0] on every top-level category, so `_SCAN_MAX_ITERATIONS`
    silently re-granted a full budget per category. With the fix, one
    counter object is threaded through every category's recursion, so a
    category that exhausts the shared budget starves every category that
    comes after it in iteration order — even one with plenty of its own
    (well under max_per_category) items.
    """
    browser = loaded_browser
    monkeypatch.setattr(browser, "_SCAN_MAX_ITERATIONS", 5)
    fake = _FakeBrowser(
        # First category (dict insertion order): 10 children — enough
        # alone to blow the shared 5-iteration budget.
        instruments=_flat_folder("instruments", 10),
        # Second category: only 3 children — trivially fits under BOTH
        # max_per_category (default) and the 5-iteration budget in
        # isolation. Under the old per-category-reset bug this would come
        # back with all 3 items; under the fix it must come back empty
        # because the shared counter is already exhausted.
        audio_effects=_flat_folder("audio_effects", 3),
    )
    monkeypatch.setattr(browser, "_get_browser", lambda: fake)

    result = browser.scan_browser_deep(None, {})

    assert result["counts"]["instruments"] == 5
    assert result["category_truncated"]["instruments"] is True

    # The crux of the regression test: category 2 must be starved by
    # category 1's consumption of the GLOBAL budget.
    assert result["categories"]["audio_effects"] == []
    assert result["counts"]["audio_effects"] == 0
    # Flagged truncated too — its incomplete (empty) result is a lower
    # bound, not "this category is genuinely empty".
    assert result["category_truncated"]["audio_effects"] is True


def test_counts_and_truncation_map_cover_every_category(loaded_browser, monkeypatch):
    """`counts` and `category_truncated` must be present for every
    category the browser exposes, not just the ones that got items."""
    browser = loaded_browser
    fake = _FakeBrowser(
        instruments=_flat_folder("instruments", 3),
        packs=_flat_folder("packs", 0),
    )
    monkeypatch.setattr(browser, "_get_browser", lambda: fake)

    result = browser.scan_browser_deep(None, {"max_per_category": 100})

    for cat in _FakeBrowser._BASE_CATEGORIES:
        assert cat in result["categories"]
        assert cat in result["counts"]
        assert cat in result["category_truncated"]

    assert result["counts"]["instruments"] == 3
    assert result["category_truncated"]["instruments"] is False
    assert result["counts"]["packs"] == 0
    assert result["category_truncated"]["packs"] is False

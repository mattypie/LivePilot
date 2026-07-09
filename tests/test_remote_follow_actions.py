"""Handler-level tests for remote_script.LivePilot.follow_actions.

Covers clip follow-action get/set/clear/preset round-trip and scene
follow-action get/set/clear, plus the version-gate RuntimeError path
and out-of-range validation for chance/time/multiplier.

follow_actions.py imports `.version_detect` lazily inside each handler
(`from .version_detect import has_feature`), so we load a real
version_detect module with `Live` stubbed and pin its cached version so
has_feature() resolves deterministically without touching a live app.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = ROOT / "remote_script" / "LivePilot"


def _load_remote_follow_actions(live_version=(12, 4, 0)):
    for name in [
        "remote_script.LivePilot.follow_actions",
        "remote_script.LivePilot.version_detect",
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
    version_detect = _load(
        "remote_script.LivePilot.version_detect", REMOTE_ROOT / "version_detect.py"
    )
    # Pin the cached version so has_feature() is deterministic without a
    # real Live.Application instance.
    version_detect._cached_version = live_version
    follow_actions = _load(
        "remote_script.LivePilot.follow_actions", REMOTE_ROOT / "follow_actions.py"
    )
    return router, follow_actions, version_detect


class _FakeClip:
    def __init__(self):
        self.follow_action_enabled = False
        self.follow_action_a = 0  # stop
        self.follow_action_b = 0  # stop
        self.follow_action_chance_a = 1.0
        self.follow_action_chance_b = 0.0
        self.follow_action_time = 1.0


class _FakeScene:
    def __init__(self):
        self.follow_action_enabled = False
        self.follow_action_time = 4.0
        self.follow_action_linked = False
        self.follow_action_multiplier = 1


def _song_with_clip(clip):
    slot = types.SimpleNamespace(clip=clip)
    track = types.SimpleNamespace(clip_slots=[slot])
    return types.SimpleNamespace(tracks=[track], return_tracks=[], master_track=None)


def _song_with_scene(scene):
    return types.SimpleNamespace(scenes=[scene])


def test_set_and_get_clip_follow_action_round_trip():
    _router, fa, _vd = _load_remote_follow_actions()
    clip = _FakeClip()
    song = _song_with_clip(clip)

    result = fa.set_clip_follow_action(song, {
        "track_index": 0, "clip_index": 0,
        "action_a": "next", "action_b": "previous",
        "chance_a": 0.7, "chance_b": 0.3, "time": 2.0, "enabled": True,
    })
    assert result["action_a"] == "next"
    assert result["action_b"] == "previous"
    assert result["chance_a"] == pytest.approx(0.7)
    assert result["enabled"] is True

    readback = fa.get_clip_follow_action(song, {"track_index": 0, "clip_index": 0})
    assert readback == result


def test_set_clip_follow_action_partial_update_preserves_other_fields():
    _router, fa, _vd = _load_remote_follow_actions()
    clip = _FakeClip()
    clip.follow_action_a = 3  # next
    clip.follow_action_time = 5.0
    song = _song_with_clip(clip)

    result = fa.set_clip_follow_action(song, {
        "track_index": 0, "clip_index": 0, "chance_a": 0.9,
    })
    # chance_a updated, action_a/time untouched
    assert result["chance_a"] == pytest.approx(0.9)
    assert result["action_a"] == "next"
    assert result["time"] == pytest.approx(5.0)


def test_set_clip_follow_action_rejects_out_of_range_chance():
    _router, fa, _vd = _load_remote_follow_actions()
    clip = _FakeClip()
    song = _song_with_clip(clip)

    with pytest.raises(ValueError, match="chance_a must be"):
        fa.set_clip_follow_action(song, {
            "track_index": 0, "clip_index": 0, "chance_a": 1.5,
        })


def test_clear_clip_follow_action_disables():
    _router, fa, _vd = _load_remote_follow_actions()
    clip = _FakeClip()
    clip.follow_action_enabled = True
    song = _song_with_clip(clip)

    result = fa.clear_clip_follow_action(song, {"track_index": 0, "clip_index": 0})
    assert result == {"enabled": False}
    assert clip.follow_action_enabled is False


def test_apply_follow_action_preset_sets_all_fields():
    _router, fa, _vd = _load_remote_follow_actions()
    clip = _FakeClip()
    song = _song_with_clip(clip)

    result = fa.apply_follow_action_preset(song, {
        "track_index": 0, "clip_index": 0, "preset": "random_walk",
    })
    assert result["action_a"] == "next"
    assert result["action_b"] == "previous"
    assert result["chance_a"] == pytest.approx(0.5)
    assert result["chance_b"] == pytest.approx(0.5)
    assert result["enabled"] is True


def test_apply_follow_action_preset_unknown_name_raises():
    _router, fa, _vd = _load_remote_follow_actions()
    clip = _FakeClip()
    song = _song_with_clip(clip)

    with pytest.raises(ValueError, match="Unknown preset"):
        fa.apply_follow_action_preset(song, {
            "track_index": 0, "clip_index": 0, "preset": "nonexistent_preset",
        })


def test_clip_follow_action_requires_live_12_0():
    """Below the clip_follow_action_v2 gate (12.0.0) every handler raises
    RuntimeError regardless of clip state — routed by dispatch to STATE_ERROR."""
    router, fa, _vd = _load_remote_follow_actions(live_version=(11, 3, 0))
    clip = _FakeClip()
    song = _song_with_clip(clip)

    response = router.dispatch(song, {
        "id": "1", "type": "get_clip_follow_action",
        "params": {"track_index": 0, "clip_index": 0},
    })
    assert response["ok"] is False
    assert response["error"]["code"] == "STATE_ERROR"
    assert "Live 12.0" in response["error"]["message"]


def test_scene_follow_action_round_trip():
    _router, fa, _vd = _load_remote_follow_actions()
    scene = _FakeScene()
    song = _song_with_scene(scene)

    result = fa.set_scene_follow_action(song, {
        "scene_index": 0, "enabled": True, "time": 8.0,
        "linked": True, "multiplier": 4,
    })
    assert result["enabled"] is True
    assert result["time"] == pytest.approx(8.0)
    assert result["linked"] is True
    assert result["multiplier"] == 4

    readback = fa.get_scene_follow_action(song, {"scene_index": 0})
    assert readback == result


def test_set_scene_follow_action_rejects_out_of_range_multiplier():
    _router, fa, _vd = _load_remote_follow_actions()
    scene = _FakeScene()
    song = _song_with_scene(scene)

    with pytest.raises(ValueError, match="multiplier must be"):
        fa.set_scene_follow_action(song, {"scene_index": 0, "multiplier": 20})


def test_clear_scene_follow_action_disables():
    _router, fa, _vd = _load_remote_follow_actions()
    scene = _FakeScene()
    scene.follow_action_enabled = True
    song = _song_with_scene(scene)

    result = fa.clear_scene_follow_action(song, {"scene_index": 0})
    assert result == {"enabled": False}
    assert scene.follow_action_enabled is False


def test_get_scene_follow_action_scene_index_out_of_range_raises_index_error():
    _router, fa, _vd = _load_remote_follow_actions()
    song = _song_with_scene(_FakeScene())

    with pytest.raises(IndexError):
        fa.get_scene_follow_action(song, {"scene_index": 5})


def test_list_follow_action_types_returns_all_nine_names():
    _router, fa, _vd = _load_remote_follow_actions()
    result = fa.list_follow_action_types(None, {})
    assert result["actions"] == [
        "stop", "play_again", "previous", "next", "first",
        "last", "any", "other", "jump",
    ]

"""Handler-level tests for remote_script.LivePilot.scenes.

Covers: get_scenes_info tempo-zero-means-None convention, create/delete/
duplicate/fire scene, set_scene_name/color/tempo (with tempo range
validation), the scene x track matrix builder (get_scene_matrix), scoped
fire_scene_clips, stop_all_clips, and get_playing_clips.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = ROOT / "remote_script" / "LivePilot"


def _load_remote_scenes():
    for name in [
        "remote_script.LivePilot.scenes",
        "remote_script.LivePilot.router",
        "remote_script.LivePilot.utils",
        "remote_script.LivePilot",
        "remote_script",
    ]:
        sys.modules.pop(name, None)

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
    scenes = _load("remote_script.LivePilot.scenes", REMOTE_ROOT / "scenes.py")
    return router, scenes


class _FakeScene:
    def __init__(self, name="Scene", tempo=-1.0, color_index=0):
        self.name = name
        self.tempo = tempo
        self.color_index = color_index
        self.fired = False

    def fire(self):
        self.fired = True


class _FakeClip:
    def __init__(self, name="Clip", is_playing=False, is_triggered=False,
                 is_recording=False, color_index=0):
        self.name = name
        self.is_playing = is_playing
        self.is_triggered = is_triggered
        self.is_recording = is_recording
        self.color_index = color_index


class _FakeSlot:
    def __init__(self, clip=None):
        self.clip = clip
        self.has_clip = clip is not None
        self.fired = False

    def fire(self):
        self.fired = True


class _FakeTrack:
    def __init__(self, name, clip_slots):
        self.name = name
        self.clip_slots = clip_slots


class _FakeSong:
    def __init__(self, scenes, tracks=None):
        self._scenes = list(scenes)
        self.tracks = list(tracks) if tracks is not None else []
        self.stopped = False

    @property
    def scenes(self):
        return self._scenes

    def create_scene(self, index):
        new_scene = _FakeScene(name="Scene %d" % (len(self._scenes) + 1))
        if index == -1:
            self._scenes.append(new_scene)
        else:
            self._scenes.insert(index, new_scene)

    def delete_scene(self, index):
        del self._scenes[index]

    def duplicate_scene(self, index):
        src = self._scenes[index]
        clone = _FakeScene(name=src.name + " Copy", tempo=src.tempo,
                            color_index=src.color_index)
        self._scenes.insert(index + 1, clone)

    def stop_all_clips(self):
        self.stopped = True


def test_get_scenes_info_reports_none_tempo_when_unset():
    _router, scenes_mod = _load_remote_scenes()
    song = _FakeSong([_FakeScene(name="Intro", tempo=-1.0), _FakeScene(name="Drop", tempo=140.0)])

    result = scenes_mod.get_scenes_info(song, {})
    assert result["scenes"][0]["tempo"] is None
    assert result["scenes"][1]["tempo"] == 140.0


def test_create_scene_appends_at_end_by_default():
    _router, scenes_mod = _load_remote_scenes()
    song = _FakeSong([_FakeScene(name="A")])

    result = scenes_mod.create_scene(song, {})
    assert result["index"] == 1
    assert len(song.scenes) == 2


def test_delete_scene_removes_and_reports_deleted_index():
    _router, scenes_mod = _load_remote_scenes()
    song = _FakeSong([_FakeScene(name="A"), _FakeScene(name="B")])

    result = scenes_mod.delete_scene(song, {"scene_index": 0})
    assert result == {"deleted": 0}
    assert len(song.scenes) == 1
    assert song.scenes[0].name == "B"


def test_delete_scene_out_of_range_raises_index_error():
    _router, scenes_mod = _load_remote_scenes()
    song = _FakeSong([_FakeScene(name="A")])

    with pytest.raises(IndexError):
        scenes_mod.delete_scene(song, {"scene_index": 9})


def test_duplicate_scene_inserts_copy_after_source():
    _router, scenes_mod = _load_remote_scenes()
    song = _FakeSong([_FakeScene(name="A"), _FakeScene(name="B")])

    result = scenes_mod.duplicate_scene(song, {"scene_index": 0})
    assert result["index"] == 1
    assert song.scenes[1].name == "A Copy"
    assert len(song.scenes) == 3


def test_fire_scene_marks_scene_fired():
    _router, scenes_mod = _load_remote_scenes()
    scene = _FakeScene(name="A")
    song = _FakeSong([scene])

    result = scenes_mod.fire_scene(song, {"scene_index": 0})
    assert result == {"index": 0, "fired": True}
    assert scene.fired is True


def test_set_scene_name_and_color():
    _router, scenes_mod = _load_remote_scenes()
    song = _FakeSong([_FakeScene(name="Old")])

    name_result = scenes_mod.set_scene_name(song, {"scene_index": 0, "name": "New"})
    assert name_result == {"index": 0, "name": "New"}

    color_result = scenes_mod.set_scene_color(song, {"scene_index": 0, "color_index": 5})
    assert color_result == {"index": 0, "color_index": 5}


def test_set_scene_tempo_validates_range():
    _router, scenes_mod = _load_remote_scenes()
    song = _FakeSong([_FakeScene(name="A")])

    result = scenes_mod.set_scene_tempo(song, {"scene_index": 0, "tempo": 128.0})
    assert result == {"index": 0, "tempo": 128.0}

    with pytest.raises(ValueError, match="between 20 and 999"):
        scenes_mod.set_scene_tempo(song, {"scene_index": 0, "tempo": 5.0})
    with pytest.raises(ValueError, match="between 20 and 999"):
        scenes_mod.set_scene_tempo(song, {"scene_index": 0, "tempo": 1200.0})


def test_get_scene_matrix_reports_cell_states():
    _router, scenes_mod = _load_remote_scenes()
    playing_clip = _FakeClip(name="Loop", is_playing=True)
    stopped_clip = _FakeClip(name="Idle")
    track_a = _FakeTrack("Drums", [_FakeSlot(playing_clip)])
    track_b = _FakeTrack("Bass", [_FakeSlot(stopped_clip)])
    song = _FakeSong([_FakeScene(name="A")], tracks=[track_a, track_b])

    result = scenes_mod.get_scene_matrix(song, {})
    assert result["tracks"] == [{"index": 0, "name": "Drums"}, {"index": 1, "name": "Bass"}]
    row = result["matrix"][0]
    assert row[0]["state"] == "playing"
    assert row[0]["name"] == "Loop"
    assert row[1]["state"] == "stopped"


def test_get_scene_matrix_reports_missing_when_track_has_fewer_slots():
    _router, scenes_mod = _load_remote_scenes()
    track_short = _FakeTrack("Short", [])  # no slots at all
    song = _FakeSong([_FakeScene(name="A")], tracks=[track_short])

    result = scenes_mod.get_scene_matrix(song, {})
    assert result["matrix"][0][0]["state"] == "missing"


def test_fire_scene_clips_all_when_no_track_filter():
    _router, scenes_mod = _load_remote_scenes()
    scene = _FakeScene(name="A")
    song = _FakeSong([scene])

    result = scenes_mod.fire_scene_clips(song, {"scene_index": 0})
    assert result == {"scene_index": 0, "fired": "all"}
    assert scene.fired is True


def test_fire_scene_clips_scoped_to_track_indices():
    _router, scenes_mod = _load_remote_scenes()
    slot_a = _FakeSlot(_FakeClip())
    slot_b = _FakeSlot(_FakeClip())
    track_a = _FakeTrack("A", [slot_a])
    track_b = _FakeTrack("B", [slot_b])
    song = _FakeSong([_FakeScene(name="S")], tracks=[track_a, track_b])

    result = scenes_mod.fire_scene_clips(song, {
        "scene_index": 0, "track_indices": [1],
    })
    assert result == {"scene_index": 0, "fired_tracks": [1]}
    assert slot_a.fired is False
    assert slot_b.fired is True


def test_fire_scene_clips_invalid_track_index_raises():
    _router, scenes_mod = _load_remote_scenes()
    song = _FakeSong([_FakeScene(name="S")], tracks=[_FakeTrack("A", [])])

    with pytest.raises(IndexError):
        scenes_mod.fire_scene_clips(song, {
            "scene_index": 0, "track_indices": [5],
        })


def test_stop_all_clips_delegates_to_song():
    _router, scenes_mod = _load_remote_scenes()
    song = _FakeSong([])

    result = scenes_mod.stop_all_clips(song, {})
    assert result == {"stopped": True}
    assert song.stopped is True


def test_get_playing_clips_filters_to_playing_or_triggered():
    _router, scenes_mod = _load_remote_scenes()
    playing = _FakeClip(name="Playing", is_playing=True)
    triggered = _FakeClip(name="Triggered", is_triggered=True)
    idle = _FakeClip(name="Idle")
    track = _FakeTrack("T", [_FakeSlot(playing), _FakeSlot(triggered), _FakeSlot(idle)])
    song = _FakeSong([], tracks=[track])

    result = scenes_mod.get_playing_clips(song, {})
    names = {c["clip_name"] for c in result["clips"]}
    assert names == {"Playing", "Triggered"}

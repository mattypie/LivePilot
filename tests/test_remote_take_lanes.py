"""Handler-level tests for remote_script.LivePilot.take_lanes.

Covers: read-only introspection (get_take_lanes / get_take_lane_clips)
that must work even when the mutation API isn't available, the
mutation ops gated behind take_lanes_api (12.2+), lane_index validation,
start_time/length validation for clip creation, and the "method not
present on this Live build" defensive RuntimeError paths.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = ROOT / "remote_script" / "LivePilot"


def _load_remote_take_lanes(live_version=(12, 4, 0)):
    for name in [
        "remote_script.LivePilot.take_lanes",
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
    version_detect._cached_version = live_version
    take_lanes = _load(
        "remote_script.LivePilot.take_lanes", REMOTE_ROOT / "take_lanes.py"
    )
    return router, take_lanes, version_detect


class _FakeArrClip:
    def __init__(self, name, start_time, length, is_midi_clip=False):
        self.name = name
        self.start_time = start_time
        self.length = length
        self.is_midi_clip = is_midi_clip


class _FakeTakeLane:
    def __init__(self, name="Take 1", is_frozen=False, clips=None,
                 support_audio=True, support_midi=True):
        self.name = name
        self.is_frozen = is_frozen
        self.clips = list(clips) if clips is not None else []
        if support_audio:
            self.create_audio_clip = self._create_audio_clip
        if support_midi:
            self.create_midi_clip = self._create_midi_clip

    def _create_audio_clip(self, start, length):
        self.clips.append(_FakeArrClip("audio", start, length, is_midi_clip=False))

    def _create_midi_clip(self, start, length):
        self.clips.append(_FakeArrClip("midi", start, length, is_midi_clip=True))


class _FakeTrack:
    def __init__(self, take_lanes=None, support_create=True):
        self.take_lanes = list(take_lanes) if take_lanes is not None else []
        if support_create:
            self.create_take_lane = self._create_take_lane

    def _create_take_lane(self):
        self.take_lanes.append(_FakeTakeLane(name="Take %d" % (len(self.take_lanes) + 1)))


def _song_with_track(track):
    return types.SimpleNamespace(tracks=[track], return_tracks=[], master_track=None)


def test_get_take_lanes_lists_all_with_clip_counts():
    _router, tl, _vd = _load_remote_take_lanes()
    lane = _FakeTakeLane(name="Comp A", clips=[
        _FakeArrClip("c1", 0.0, 4.0), _FakeArrClip("c2", 4.0, 4.0),
    ])
    track = _FakeTrack(take_lanes=[lane])
    song = _song_with_track(track)

    result = tl.get_take_lanes(song, {"track_index": 0})
    assert result["lanes"] == [{
        "index": 0, "name": "Comp A", "is_frozen": False, "clip_count": 2,
    }]


def test_get_take_lanes_returns_empty_when_track_has_no_take_lanes_attr():
    """A track object without a take_lanes attribute at all (older Live
    builds) must degrade to an empty list, not raise."""
    _router, tl, _vd = _load_remote_take_lanes()
    track = types.SimpleNamespace()  # no take_lanes attribute
    song = _song_with_track(track)

    result = tl.get_take_lanes(song, {"track_index": 0})
    assert result == {"lanes": []}


def test_create_take_lane_appends_and_returns_new_index():
    _router, tl, _vd = _load_remote_take_lanes()
    track = _FakeTrack(take_lanes=[_FakeTakeLane(name="Existing")])
    song = _song_with_track(track)

    result = tl.create_take_lane(song, {"track_index": 0})
    assert result["lane_index"] == 1
    assert result["name"] == "Take 2"
    assert len(track.take_lanes) == 2


def test_create_take_lane_missing_method_raises_runtime_error():
    _router, tl, _vd = _load_remote_take_lanes()
    track = _FakeTrack(support_create=False)
    song = _song_with_track(track)

    with pytest.raises(RuntimeError, match="not available"):
        tl.create_take_lane(song, {"track_index": 0})


def test_create_take_lane_requires_live_12_2():
    router, _tl, _vd = _load_remote_take_lanes(live_version=(12, 1, 0))
    track = _FakeTrack()
    song = _song_with_track(track)

    response = router.dispatch(song, {
        "id": "1", "type": "create_take_lane", "params": {"track_index": 0},
    })
    assert response["ok"] is False
    assert response["error"]["code"] == "STATE_ERROR"
    assert "Live 12.2" in response["error"]["message"]


def test_set_take_lane_name_renames():
    _router, tl, _vd = _load_remote_take_lanes()
    lane = _FakeTakeLane(name="Old")
    track = _FakeTrack(take_lanes=[lane])
    song = _song_with_track(track)

    result = tl.set_take_lane_name(song, {
        "track_index": 0, "lane_index": 0, "name": "New",
    })
    assert result == {"name": "New"}
    assert lane.name == "New"


def test_set_take_lane_name_invalid_lane_index_raises_index_error():
    _router, tl, _vd = _load_remote_take_lanes()
    track = _FakeTrack(take_lanes=[_FakeTakeLane()])
    song = _song_with_track(track)

    with pytest.raises(IndexError, match="out of range"):
        tl.set_take_lane_name(song, {
            "track_index": 0, "lane_index": 3, "name": "New",
        })


def test_create_audio_clip_on_take_lane_happy_path():
    _router, tl, _vd = _load_remote_take_lanes()
    lane = _FakeTakeLane()
    track = _FakeTrack(take_lanes=[lane])
    song = _song_with_track(track)

    result = tl.create_audio_clip_on_take_lane(song, {
        "track_index": 0, "lane_index": 0, "start_time": 8.0, "length": 4.0,
    })
    assert result["ok"] is True
    assert result["start_time"] == 8.0
    assert len(lane.clips) == 1
    assert lane.clips[0].is_midi_clip is False


def test_create_audio_clip_on_take_lane_rejects_non_positive_length():
    _router, tl, _vd = _load_remote_take_lanes()
    lane = _FakeTakeLane()
    track = _FakeTrack(take_lanes=[lane])
    song = _song_with_track(track)

    with pytest.raises(ValueError, match="length must be > 0"):
        tl.create_audio_clip_on_take_lane(song, {
            "track_index": 0, "lane_index": 0, "start_time": 0.0, "length": 0.0,
        })


def test_create_audio_clip_on_take_lane_missing_method_raises():
    _router, tl, _vd = _load_remote_take_lanes()
    lane = _FakeTakeLane(support_audio=False)
    track = _FakeTrack(take_lanes=[lane])
    song = _song_with_track(track)

    with pytest.raises(RuntimeError, match="not available"):
        tl.create_audio_clip_on_take_lane(song, {
            "track_index": 0, "lane_index": 0, "start_time": 0.0, "length": 4.0,
        })


def test_create_midi_clip_on_take_lane_happy_path():
    _router, tl, _vd = _load_remote_take_lanes()
    lane = _FakeTakeLane()
    track = _FakeTrack(take_lanes=[lane])
    song = _song_with_track(track)

    result = tl.create_midi_clip_on_take_lane(song, {
        "track_index": 0, "lane_index": 0, "start_time": 0.0, "length": 8.0,
    })
    assert result["ok"] is True
    assert lane.clips[0].is_midi_clip is True


def test_get_take_lane_clips_lists_arrangement_clips():
    _router, tl, _vd = _load_remote_take_lanes()
    lane = _FakeTakeLane(clips=[
        _FakeArrClip("Take 1", 0.0, 4.0, is_midi_clip=True),
        _FakeArrClip("Take 2", 4.0, 8.0, is_midi_clip=False),
    ])
    track = _FakeTrack(take_lanes=[lane])
    song = _song_with_track(track)

    result = tl.get_take_lane_clips(song, {"track_index": 0, "lane_index": 0})
    assert len(result["clips"]) == 2
    assert result["clips"][0]["is_midi_clip"] is True
    assert result["clips"][1]["length"] == 8.0


def test_get_take_lane_clips_is_not_version_gated():
    """Read-only introspection must work below the 12.2 mutation gate."""
    _router, tl, _vd = _load_remote_take_lanes(live_version=(12, 0, 0))
    lane = _FakeTakeLane(clips=[_FakeArrClip("Take 1", 0.0, 4.0)])
    track = _FakeTrack(take_lanes=[lane])
    song = _song_with_track(track)

    result = tl.get_take_lane_clips(song, {"track_index": 0, "lane_index": 0})
    assert len(result["clips"]) == 1

"""Contract diff: fixtures_remote.py canonical shapes vs the real handlers.

fixtures_remote.py is hand-seeded from reading tracks.py/transport.py/
devices.py/scenes.py. Hand-seeding drifts silently the moment a handler
adds/removes/renames a key. This test closes that loop: it loads the real
handler modules via the same isolated-import harness as
test_remote_script_contracts.py, executes each handler against a
realistic fake-LOM object, and asserts the resulting key set matches what
fixtures_remote.py's builder produces (top-level keys, plus the nested
per-row shapes for list-valued fields).

If a handler changes shape, this test fails — that's the point. Update
fixtures_remote.py to match, don't relax this test's key-set assertions.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

from tests.fixtures_remote import (
    make_batch_set_parameters_result,
    make_device_parameters,
    make_scene_matrix,
    make_session_info,
    make_track_clip_slot,
    make_track_device,
    make_track_info,
)


ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = ROOT / "remote_script" / "LivePilot"


def _reset_remote_modules(names):
    for name in names:
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


def _load_tracks():
    _reset_remote_modules([
        "remote_script.LivePilot.tracks",
        "remote_script.LivePilot.router",
        "remote_script.LivePilot.utils",
        "remote_script.LivePilot",
        "remote_script",
    ])
    _load("remote_script.LivePilot.utils", REMOTE_ROOT / "utils.py")
    _load("remote_script.LivePilot.router", REMOTE_ROOT / "router.py")
    return _load("remote_script.LivePilot.tracks", REMOTE_ROOT / "tracks.py")


def _load_transport():
    _reset_remote_modules([
        "remote_script.LivePilot.transport",
        "remote_script.LivePilot.version_detect",
        "remote_script.LivePilot.router",
        "remote_script.LivePilot.utils",
        "remote_script.LivePilot",
        "remote_script",
    ])
    sys.modules.setdefault("Live", types.ModuleType("Live"))
    _load("remote_script.LivePilot.utils", REMOTE_ROOT / "utils.py")
    _load("remote_script.LivePilot.router", REMOTE_ROOT / "router.py")
    _load("remote_script.LivePilot.version_detect", REMOTE_ROOT / "version_detect.py")
    return _load("remote_script.LivePilot.transport", REMOTE_ROOT / "transport.py")


def _load_devices():
    _reset_remote_modules([
        "remote_script.LivePilot.devices",
        "remote_script.LivePilot.router",
        "remote_script.LivePilot.utils",
        "remote_script.LivePilot",
        "remote_script",
    ])
    sys.modules.setdefault("Live", types.ModuleType("Live"))
    _load("remote_script.LivePilot.utils", REMOTE_ROOT / "utils.py")
    _load("remote_script.LivePilot.router", REMOTE_ROOT / "router.py")
    return _load("remote_script.LivePilot.devices", REMOTE_ROOT / "devices.py")


def _load_scenes():
    _reset_remote_modules([
        "remote_script.LivePilot.scenes",
        "remote_script.LivePilot.router",
        "remote_script.LivePilot.utils",
        "remote_script.LivePilot",
        "remote_script",
    ])
    _load("remote_script.LivePilot.utils", REMOTE_ROOT / "utils.py")
    _load("remote_script.LivePilot.router", REMOTE_ROOT / "router.py")
    return _load("remote_script.LivePilot.scenes", REMOTE_ROOT / "scenes.py")


# ─── Minimal fake-LOM objects ────────────────────────────────────────────────

class _Param:
    def __init__(self, name="Cutoff", value=0.5, min=0.0, max=1.0, is_quantized=False):
        self.name = name
        self.value = value
        self.min = min
        self.max = max
        self.is_quantized = is_quantized

    def str_for_value(self, v):
        return "%.2f" % v

    @property
    def display_value(self):
        return "%.2f" % self.value


class _Device:
    def __init__(self, name="Drift", class_name="InstrumentGroupDevice", is_active=True, parameters=None):
        self.name = name
        self.class_name = class_name
        self.is_active = is_active
        self.parameters = parameters if parameters is not None else [_Param()]


class _Clip:
    name = "Clip"
    color_index = 0
    length = 4.0
    is_playing = False
    is_recording = False
    is_triggered = False
    looping = True
    loop_start = 0.0
    loop_end = 4.0
    start_marker = 0.0
    end_marker = 4.0


class _Slot:
    def __init__(self, has_clip=False, clip=None):
        self.has_clip = has_clip
        self.clip = clip


class _Mixer:
    class _P:
        def __init__(self, value):
            self.value = value

    def __init__(self, volume=0.85, panning=0.0):
        self.volume = self._P(volume)
        self.panning = self._P(panning)
        self.sends = []


class _Track:
    def __init__(self, index=0, name="Track 1", devices=None, slots=None):
        self.name = name
        self.color_index = 0
        self.mute = False
        self.solo = False
        self.is_foldable = False
        self.is_grouped = False
        self.clip_slots = slots if slots is not None else [_Slot()]
        self.devices = devices if devices is not None else []
        self.mixer_device = _Mixer()
        self.arm = False
        self.has_midi_input = True
        self.has_audio_input = False
        self.current_monitoring_state = 0


class _Scene:
    def __init__(self, name="Scene 1"):
        self.name = name
        self.color_index = 0
        self.tempo = 0.0  # 0 -> None in the real handler


class _Song:
    def __init__(self, tracks=None, return_tracks=None, scenes=None):
        self.tracks = tracks if tracks is not None else [_Track()]
        self.return_tracks = return_tracks if return_tracks is not None else []
        self.scenes = scenes if scenes is not None else [_Scene()]
        self.tempo = 120.0
        self.signature_numerator = 4
        self.signature_denominator = 4
        self.is_playing = False
        self.song_length = 64.0
        self.current_song_time = 0.0
        self.loop = False
        self.loop_start = 0.0
        self.loop_length = 16.0
        self.metronome = False
        self.record_mode = False
        self.session_record = False


def _keys(d):
    return set(d.keys())


def test_get_track_info_keys_match_fixture():
    tracks = _load_tracks()
    song = _Song()
    real = tracks.get_track_info(song, {"track_index": 0})
    fixture = make_track_info(0)
    assert _keys(real) == _keys(fixture), (
        f"get_track_info handler shape drifted from fixtures_remote.make_track_info: "
        f"real={_keys(real)} fixture={_keys(fixture)}"
    )
    # Nested per-clip-slot shape (non-empty slot) must line up too.
    clip_track = _Track(slots=[_Slot(has_clip=True, clip=_Clip())])
    real_with_clip = tracks.get_track_info(_Song(tracks=[clip_track]), {"track_index": 0})
    assert _keys(real_with_clip["clip_slots"][0]) == _keys(make_track_clip_slot(0, has_clip=True))
    # Nested per-device shape.
    dev_track = _Track(devices=[_Device()])
    real_with_device = tracks.get_track_info(_Song(tracks=[dev_track]), {"track_index": 0})
    assert _keys(real_with_device["devices"][0]) == _keys(make_track_device())


def test_get_session_info_keys_match_fixture():
    transport = _load_transport()
    real = transport.get_session_info(_Song(), {})
    fixture = make_session_info()
    assert _keys(real) == _keys(fixture), (
        f"get_session_info handler shape drifted from fixtures_remote.make_session_info: "
        f"real={_keys(real)} fixture={_keys(fixture)}"
    )
    from tests.fixtures_remote import make_session_track_summary, make_session_scene_summary
    assert _keys(real["tracks"][0]) == _keys(make_session_track_summary())
    assert _keys(real["scenes"][0]) == _keys(make_session_scene_summary())


def test_get_device_parameters_keys_match_fixture():
    devices = _load_devices()
    song = _Song(tracks=[_Track(devices=[_Device()])])
    real = devices.get_device_parameters(song, {"track_index": 0, "device_index": 0})
    fixture = make_device_parameters()
    assert _keys(real) == _keys(fixture)
    assert _keys(real["parameters"][0]) == _keys(fixture["parameters"][0])


def test_batch_set_parameters_keys_match_fixture():
    devices = _load_devices()

    class _UndoSong(_Song):
        def begin_undo_step(self):
            pass

        def end_undo_step(self):
            pass

    song = _UndoSong(tracks=[_Track(devices=[_Device(parameters=[_Param("Cutoff", 0.4)])])])
    real = devices.batch_set_parameters(song, {
        "track_index": 0,
        "device_index": 0,
        "parameters": [{"name_or_index": "Cutoff", "value": 0.6}],
    })
    fixture = make_batch_set_parameters_result()
    assert _keys(real) == _keys(fixture)
    assert _keys(real["parameters"][0]) == _keys(fixture["parameters"][0])


def test_get_scene_matrix_keys_match_fixture():
    scenes = _load_scenes()
    real = scenes.get_scene_matrix(_Song(), {})
    fixture = make_scene_matrix()
    assert _keys(real) == _keys(fixture)
    assert _keys(real["tracks"][0]) == _keys(fixture["tracks"][0])
    assert _keys(real["scenes"][0]) == _keys(fixture["scenes"][0])
    assert _keys(real["matrix"][0][0]) == _keys(fixture["matrix"][0][0])

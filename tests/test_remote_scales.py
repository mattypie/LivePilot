"""Handler-level tests for remote_script.LivePilot.scales.

Covers: song-scale get/set (root note as int or note name, case-insensitive
scale-name match), scale-mode toggle, the 12.4 scale_names-drop fallback
(_resolve_scale_names/_BUILTIN_SCALES_FALLBACK), and the tuning-system
handlers (read, reference pitch, per-degree cents, reset).

scales.py imports `.version_detect` lazily inside each handler, so this
loader mirrors test_remote_follow_actions.py's pattern: load a real
version_detect with Live stubbed and pin _cached_version.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = ROOT / "remote_script" / "LivePilot"


def _load_remote_scales(live_version=(12, 4, 0)):
    for name in [
        "remote_script.LivePilot.scales",
        "remote_script.LivePilot._scale_helpers",
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
    _load("remote_script.LivePilot._scale_helpers", REMOTE_ROOT / "_scale_helpers.py")
    scales = _load("remote_script.LivePilot.scales", REMOTE_ROOT / "scales.py")
    return router, scales, version_detect


class _FakeTuningSystem:
    def __init__(self):
        self.name = "12-TET"
        self.pseudo_octave_in_cents = 1200.0
        self.lowest_note = 0
        self.highest_note = 127
        self.reference_pitch = 440.0
        self.note_tunings = [0.0] * 12


class _FakeSong:
    def __init__(self, scale_names=None):
        self.root_note = 0
        self.scale_mode = False
        self.scale_name = "Major"
        self.scale_intervals = [0, 2, 4, 5, 7, 9, 11]
        if scale_names is not None:
            self.scale_names = scale_names
        self.tuning_system = _FakeTuningSystem()


def test_get_song_scale_returns_full_state():
    _router, scales, _vd = _load_remote_scales()
    song = _FakeSong(scale_names=["Major", "Minor", "Dorian"])

    result = scales.get_song_scale(song, {})
    assert result["root_note"] == 0
    assert result["scale_mode"] is False
    assert result["scale_name"] == "Major"
    assert result["available_scales"] == ["Major", "Minor", "Dorian"]


def test_set_song_scale_accepts_note_name_string():
    _router, scales, _vd = _load_remote_scales()
    song = _FakeSong(scale_names=["Major", "Minor"])

    result = scales.set_song_scale(song, {"root_note": "F#", "scale_name": "minor"})
    assert result["root_note"] == 6
    assert result["scale_name"] == "Minor"  # matched against Live's exact casing


def test_set_song_scale_accepts_int_root_note():
    _router, scales, _vd = _load_remote_scales()
    song = _FakeSong(scale_names=["Major"])

    result = scales.set_song_scale(song, {"root_note": 9, "scale_name": "Major"})
    assert result["root_note"] == 9


def test_set_song_scale_unknown_scale_name_raises():
    _router, scales, _vd = _load_remote_scales()
    song = _FakeSong(scale_names=["Major", "Minor"])

    with pytest.raises(ValueError, match="Unknown scale"):
        scales.set_song_scale(song, {"root_note": 0, "scale_name": "Bebop"})


def test_set_song_scale_rejects_invalid_root_note_string():
    _router, scales, _vd = _load_remote_scales()
    song = _FakeSong(scale_names=["Major"])

    with pytest.raises(ValueError, match="Unknown note name"):
        scales.set_song_scale(song, {"root_note": "H", "scale_name": "Major"})


def test_set_song_scale_rejects_bool_root_note():
    _router, scales, _vd = _load_remote_scales()
    song = _FakeSong(scale_names=["Major"])

    with pytest.raises(ValueError, match="cannot be a boolean"):
        scales.set_song_scale(song, {"root_note": True, "scale_name": "Major"})


def test_set_song_scale_mode_toggles():
    _router, scales, _vd = _load_remote_scales()
    song = _FakeSong()

    result = scales.set_song_scale_mode(song, {"enabled": True})
    assert result["scale_mode"] is True
    assert song.scale_mode is True


def test_list_available_scales_falls_back_when_scale_names_missing():
    """Live 12.4 dropped Song.scale_names from the Python LOM in some
    builds — list_available_scales must still return a usable list via
    the built-in fallback rather than raising AttributeError."""
    _router, scales, _vd = _load_remote_scales()
    song = _FakeSong()  # no scale_names attribute set

    result = scales.list_available_scales(song, {})
    assert "Major" in result["scales"]
    assert "Minor" in result["scales"]
    assert len(result["scales"]) > 10


def test_song_scale_requires_live_12_0():
    router, _scales, _vd_pinned_low = _load_remote_scales(live_version=(11, 3, 0))
    song = _FakeSong()

    response = router.dispatch(song, {"id": "1", "type": "get_song_scale", "params": {}})
    assert response["ok"] is False
    assert response["error"]["code"] == "STATE_ERROR"
    assert "Live 12.0" in response["error"]["message"]


def test_get_tuning_system_returns_full_state():
    _router, scales, _vd = _load_remote_scales()
    song = _FakeSong()

    result = scales.get_tuning_system(song, {})
    assert result["name"] == "12-TET"
    assert result["reference_pitch"] == pytest.approx(440.0)
    assert len(result["note_tunings"]) == 12


def test_set_tuning_reference_pitch_updates_and_rejects_non_positive():
    _router, scales, _vd = _load_remote_scales()
    song = _FakeSong()

    result = scales.set_tuning_reference_pitch(song, {"reference_pitch": 432.0})
    assert result["reference_pitch"] == pytest.approx(432.0)

    with pytest.raises(ValueError, match="must be > 0"):
        scales.set_tuning_reference_pitch(song, {"reference_pitch": 0.0})


def test_set_tuning_note_updates_single_degree():
    _router, scales, _vd = _load_remote_scales()
    song = _FakeSong()

    result = scales.set_tuning_note(song, {"degree": 3, "cent_offset": 15.5})
    assert result == {"degree": 3, "cent_offset": 15.5}
    assert song.tuning_system.note_tunings[3] == pytest.approx(15.5)
    # Other degrees untouched
    assert song.tuning_system.note_tunings[0] == pytest.approx(0.0)


def test_set_tuning_note_out_of_range_degree_raises_index_error():
    _router, scales, _vd = _load_remote_scales()
    song = _FakeSong()

    with pytest.raises(IndexError, match="out of range"):
        scales.set_tuning_note(song, {"degree": 99, "cent_offset": 5.0})


def test_reset_tuning_system_zeroes_all_offsets():
    _router, scales, _vd = _load_remote_scales()
    song = _FakeSong()
    song.tuning_system.note_tunings = [10.0] * 12

    result = scales.reset_tuning_system(song, {})
    assert result["note_tunings"] == [0.0] * 12


def test_tuning_system_requires_live_12_1():
    router, _scales, _vd = _load_remote_scales(live_version=(12, 0, 0))
    song = _FakeSong()

    response = router.dispatch(song, {"id": "1", "type": "get_tuning_system", "params": {}})
    assert response["ok"] is False
    assert response["error"]["code"] == "STATE_ERROR"
    assert "Live 12.1" in response["error"]["message"]

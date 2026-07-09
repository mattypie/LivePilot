"""Handler-level tests for remote_script.LivePilot.grooves.

Covers list/get/set groove params, clip groove assignment (including the
groove_id=-1 clear path and the "not found in current pool" identity
fallback in get_clip_groove), and the master groove_amount dial's
0.0-1.31 range (wider than the 0.0-1.0 the UI shows).
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = ROOT / "remote_script" / "LivePilot"


def _load_remote_grooves(live_version=(12, 4, 0)):
    for name in [
        "remote_script.LivePilot.grooves",
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
    grooves = _load("remote_script.LivePilot.grooves", REMOTE_ROOT / "grooves.py")
    return router, grooves, version_detect


class _FakeGroove:
    def __init__(self, name, base=0, quant=0.5, rand=0.0, timing=0.0, vel=0.0):
        self.name = name
        self.base = base
        self.quantization_amount = quant
        self.random_amount = rand
        self.timing_amount = timing
        self.velocity_amount = vel


class _FakeGroovePool:
    def __init__(self, grooves):
        self.grooves = list(grooves)


class _FakeClip:
    def __init__(self):
        self.groove = None


def _song_with_grooves(grooves, clip=None, groove_amount=1.0):
    slot = types.SimpleNamespace(clip=clip if clip is not None else _FakeClip())
    track = types.SimpleNamespace(clip_slots=[slot])
    return types.SimpleNamespace(
        tracks=[track],
        groove_pool=_FakeGroovePool(grooves),
        groove_amount=groove_amount,
    )


def test_list_grooves_returns_all_with_index_as_id():
    _router, g, _vd = _load_remote_grooves()
    song = _song_with_grooves([_FakeGroove("Swing 16", quant=0.6), _FakeGroove("MPC 60")])

    result = g.list_grooves(song, {})
    assert len(result["grooves"]) == 2
    assert result["grooves"][0]["id"] == 0
    assert result["grooves"][0]["name"] == "Swing 16"
    assert result["grooves"][1]["id"] == 1


def test_get_groove_info_out_of_range_raises_index_error():
    _router, g, _vd = _load_remote_grooves()
    song = _song_with_grooves([_FakeGroove("Solo Groove")])

    with pytest.raises(IndexError, match="out of range"):
        g.get_groove_info(song, {"groove_id": 5})


def test_set_groove_params_partial_update_and_validation():
    _router, g, _vd = _load_remote_grooves()
    groove = _FakeGroove("Groove A", quant=0.5, rand=0.1, timing=0.2, vel=-0.3)
    song = _song_with_grooves([groove])

    result = g.set_groove_params(song, {"groove_id": 0, "timing_amount": 0.8})
    assert result["timing_amount"] == pytest.approx(0.8)
    # Untouched fields preserved
    assert result["quantization_amount"] == pytest.approx(0.5)
    assert result["random_amount"] == pytest.approx(0.1)
    assert result["velocity_amount"] == pytest.approx(-0.3)


def test_set_groove_params_rejects_velocity_out_of_signed_range():
    _router, g, _vd = _load_remote_grooves()
    groove = _FakeGroove("Groove A")
    song = _song_with_grooves([groove])

    with pytest.raises(ValueError, match="velocity_amount must be"):
        g.set_groove_params(song, {"groove_id": 0, "velocity_amount": 2.0})


def test_assign_clip_groove_sets_and_clears():
    _router, g, _vd = _load_remote_grooves()
    groove = _FakeGroove("Groove A")
    clip = _FakeClip()
    song = _song_with_grooves([groove], clip=clip)

    result = g.assign_clip_groove(song, {
        "track_index": 0, "clip_index": 0, "groove_id": 0,
    })
    assert result["groove_id"] == 0
    assert result["groove_name"] == "Groove A"
    assert clip.groove is groove

    cleared = g.assign_clip_groove(song, {
        "track_index": 0, "clip_index": 0, "groove_id": -1,
    })
    assert cleared["groove_id"] is None
    assert clip.groove is None


def test_get_clip_groove_reflects_assignment_by_pool_identity():
    _router, g, _vd = _load_remote_grooves()
    groove = _FakeGroove("Groove A")
    clip = _FakeClip()
    clip.groove = groove
    song = _song_with_grooves([groove], clip=clip)

    result = g.get_clip_groove(song, {"track_index": 0, "clip_index": 0})
    assert result == {"groove_id": 0, "groove_name": "Groove A"}


def test_get_clip_groove_unset_returns_none_ids():
    _router, g, _vd = _load_remote_grooves()
    song = _song_with_grooves([_FakeGroove("Groove A")])

    result = g.get_clip_groove(song, {"track_index": 0, "clip_index": 0})
    assert result == {"groove_id": None, "groove_name": None}


def test_song_groove_amount_get_and_set_within_expanded_range():
    _router, g, _vd = _load_remote_grooves()
    song = _song_with_grooves([], groove_amount=1.0)

    result = g.set_song_groove_amount(song, {"amount": 1.31})
    assert result["groove_amount"] == pytest.approx(1.31)
    readback = g.get_song_groove_amount(song, {})
    assert readback["groove_amount"] == pytest.approx(1.31)


def test_set_song_groove_amount_rejects_over_max():
    _router, g, _vd = _load_remote_grooves()
    song = _song_with_grooves([])

    with pytest.raises(ValueError, match="amount must be"):
        g.set_song_groove_amount(song, {"amount": 2.0})


def test_groove_pool_requires_live_11():
    router, g, _vd = _load_remote_grooves(live_version=(10, 1, 0))
    song = _song_with_grooves([_FakeGroove("Groove A")])

    response = router.dispatch(song, {"id": "1", "type": "list_grooves", "params": {}})
    assert response["ok"] is False
    assert response["error"]["code"] == "STATE_ERROR"
    assert "Live 11" in response["error"]["message"]

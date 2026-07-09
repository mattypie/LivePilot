"""P1-28 regression: a write command must never dispatch after the client was
told it timed out, and a late timeout must not cancel an already-committed
write. The cancel/commit decision is serialized by a per-command lock.
"""

from __future__ import annotations

import importlib.util
import queue
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = ROOT / "remote_script" / "LivePilot"


def _load_server_module():
    for name in [
        "remote_script.LivePilot.server",
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

    def _load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    _load("remote_script.LivePilot.utils", REMOTE_ROOT / "utils.py")
    _load("remote_script.LivePilot.router", REMOTE_ROOT / "router.py")
    return _load("remote_script.LivePilot.server", REMOTE_ROOT / "server.py")


class _FakeSong(object):
    pass


class _FakeCS(object):
    def song(self):
        return _FakeSong()

    def schedule_message(self, delay, fn):
        # Execute synchronously for deterministic tests.
        fn()


def _make_server(server_mod, dispatch_calls):
    def _recording_dispatch(song, command):
        dispatch_calls.append(command)
        return {"id": command.get("id", "x"), "ok": True}

    server_mod.router.dispatch = _recording_dispatch
    return server_mod.LivePilotServer(_FakeCS())


def test_cancelled_write_is_not_dispatched():
    """Timeout won the lock first → main thread must skip dispatch entirely."""
    mod = _load_server_module()
    calls = []
    srv = _make_server(mod, calls)

    command = {"id": "1", "type": "set_track_volume", "track_index": 0, "value": 0.5}
    rq = queue.Queue()
    state = mod._CmdState()
    state.cancelled = True  # simulate the TCP thread having timed out + cancelled
    srv._command_queue.put((command, rq, state))

    srv._process_next_command()

    assert calls == [], "phantom write: cancelled command was dispatched"
    assert rq.empty(), "no response should be emitted for a cancelled command"
    assert state.committed is False


def test_committed_write_cannot_be_cancelled_by_late_timeout():
    """Main thread committed first → a late timeout must NOT flip cancelled."""
    mod = _load_server_module()
    calls = []
    srv = _make_server(mod, calls)

    command = {"id": "2", "type": "set_track_volume", "track_index": 0, "value": 0.5}
    rq = queue.Queue()
    state = mod._CmdState()
    srv._command_queue.put((command, rq, state))

    srv._process_next_command()  # commits + dispatches

    assert state.committed is True
    assert len(calls) == 1, "committed write should dispatch exactly once"

    # Now the TCP thread's timeout path runs (raced just after commit):
    with state.lock:
        if not state.committed:
            state.cancelled = True

    assert state.cancelled is False, "a committed write must not be cancellable"
    # Invariant: cancelled and committed are never both True.
    assert not (state.cancelled and state.committed)

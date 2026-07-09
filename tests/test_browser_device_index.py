"""load_browser_item must apply role-defaults / hygiene to the JUST-LOADED
device, which Live appends at the END of the chain — not to device_index 0.

Before the fix, on a non-empty track the probe + every set_device_parameter
targeted index 0 (the pre-existing device), silently mangling the wrong
instrument while the freshly loaded one kept Live's raw defaults. This is the
documented drum-Simpler Transpose=+24 silent-failure path.
"""

from __future__ import annotations

from mcp_server.tools import browser


class _FakeAbleton:
    def __init__(self, load_result):
        self._load_result = load_result
        self.param_writes = []   # list of device_index used for set_device_parameter
        self.probe_index = None  # device_index used for get_device_info

    def send_command(self, command, params=None):
        params = params or {}
        if command == "load_browser_item":
            return self._load_result
        if command == "get_device_info":
            self.probe_index = params.get("device_index")
            # The just-loaded device IS a Simpler so role-defaults should fire.
            return {"class_name": "Simpler", "name": "Kick 808"}
        if command == "set_device_parameter":
            self.param_writes.append(params.get("device_index"))
            return {"ok": True}
        return {}


class _FakeCtx:
    def __init__(self, ableton):
        self.lifespan_context = {"ableton": ableton}


def _run(load_result, role="drum"):
    ab = _FakeAbleton(load_result)
    ctx = _FakeCtx(ab)
    res = browser.load_browser_item(ctx, track_index=0, uri="query:Drums#Kick", role=role)
    return ab, res


def test_uses_remote_device_index_on_non_empty_track():
    # Track already had 2 devices; the loaded device is the 3rd (index 2).
    ab, res = _run({"loaded": True, "name": "Kick 808", "device_count": 3, "device_index": 2})
    assert ab.probe_index == 2, "must probe the appended device, not index 0"
    assert ab.param_writes, "role defaults should have been applied"
    assert set(ab.param_writes) == {2}, "all role-default writes must target the loaded device"


def test_falls_back_to_device_count_minus_one_when_index_absent():
    # Older remote that returns only device_count (no device_index).
    ab, res = _run({"loaded": True, "name": "Kick 808", "device_count": 4})
    assert ab.probe_index == 3
    assert set(ab.param_writes) == {3}


def test_empty_track_resolves_to_index_zero():
    ab, res = _run({"loaded": True, "name": "Kick 808", "device_count": 1, "device_index": 0})
    assert ab.probe_index == 0
    assert set(ab.param_writes) == {0}


def test_probe_failure_is_surfaced_not_swallowed():
    class _RaisingAbleton(_FakeAbleton):
        def send_command(self, command, params=None):
            if command == "get_device_info":
                raise ConnectionError("probe boom")
            return super().send_command(command, params)

    ab = _RaisingAbleton({"loaded": True, "name": "Kick 808", "device_count": 2, "device_index": 1})
    ctx = _FakeCtx(ab)
    res = browser.load_browser_item(ctx, track_index=0, uri="query:Drums#Kick", role="drum")
    assert "role_defaults_skipped" in res, "probe failure must be surfaced, not silent"
    assert ab.param_writes == [], "no role defaults should be claimed when probe failed"

"""Regression guard for P2-4 (DEEP_REVIEW_2026-06-24).

MidiToolCache.get_last_context / get_last_notes must derive staleness from the
request payload's own timestamp (_request_time, set only in set_request), NOT
from the shared _last_seen — which mark_ready() (the /miditool/ready heartbeat)
also bumps. Before the fix, a heartbeat arriving after an expired request reset
the staleness clock and resurrected the stale context/notes payload as if fresh.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mcp_server.m4l_bridge as m4l_bridge
from mcp_server.m4l_bridge import MidiToolCache


class _FakeClock:
    """Injectable monotonic clock so we can advance time deterministically."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _patch_clock(monkeypatch) -> _FakeClock:
    clock = _FakeClock()
    monkeypatch.setattr(m4l_bridge.time, "monotonic", clock)
    return clock


def test_heartbeat_does_not_resurrect_expired_context(monkeypatch):
    clock = _patch_clock(monkeypatch)
    cache = MidiToolCache(max_age=5.0)

    clock.now = 1000.0
    cache.set_request({"grid": 16}, [[60, 0, 1, 100, 0]])

    # Let the request payload age out past max_age.
    clock.now = 1000.0 + 6.0
    assert cache.get_last_context() is None
    assert cache.get_last_notes() is None

    # A /miditool/ready heartbeat arrives — it bumps _last_seen (is_connected)
    # but MUST NOT make the already-expired request payload look fresh again.
    cache.mark_ready()
    assert cache.is_connected is True  # heartbeat keeps the connection alive
    assert cache.get_last_context() is None  # but the stale payload stays gone
    assert cache.get_last_notes() is None


def test_fresh_request_within_max_age_is_returned(monkeypatch):
    clock = _patch_clock(monkeypatch)
    cache = MidiToolCache(max_age=5.0)

    clock.now = 2000.0
    cache.set_request({"grid": 8}, [[62, 0, 2, 90, 0]])

    clock.now = 2000.0 + 4.0  # still within max_age
    assert cache.get_last_context() == {"grid": 8}
    assert cache.get_last_notes() == [[62, 0, 2, 90, 0]]


def test_request_time_independent_of_last_seen(monkeypatch):
    """A heartbeat between set_request and the read must not extend freshness."""
    clock = _patch_clock(monkeypatch)
    cache = MidiToolCache(max_age=5.0)

    clock.now = 3000.0
    cache.set_request({"grid": 4}, [[64, 0, 1, 80, 0]])

    # Heartbeat at t+3 bumps _last_seen but not _request_time.
    clock.now = 3003.0
    cache.mark_ready()

    # At t+6 the request itself is stale even though _last_seen is only 3s old.
    clock.now = 3006.0
    assert cache.get_last_context() is None
    assert cache.get_last_notes() is None

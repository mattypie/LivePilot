"""Tests for SpliceGRPCClient connection-health tracking.

Prior bug: `connect()` only proves the local gRPC channel object
constructed — it does NOT prove Splice desktop is reachable. Real
failures surface on the first RPC. Every RPC wrapper swallowed
exceptions into benign empty defaults (empty list / False / None) and
NEVER reset `self.connected`, so one Splice restart made every later
call return fake "0 results" for the rest of the process lifetime,
indistinguishable from a genuine empty result.

These tests lock down:
  1. An RPC failure sets `degraded=True`, records `last_error`, and
     resets `connected=False` (instead of leaving it permanently True).
  2. The NEXT call after a degradation retries `connect()` exactly once
     before attempting the RPC (cheap self-healing).
  3. A client that was never connected (and never degraded) does NOT
     attempt a reconnect — matches the historical "not connected ->
     return default" behavior and avoids crashing on partially
     constructed clients (e.g. `SpliceGRPCClient.__new__` in other
     tests, which skip `__init__` entirely).
  4. `degraded` distinguishes a real RPC failure from a genuine
     zero-result response — both produce the same empty default shape,
     but only the failure flips the flag.
"""

from __future__ import annotations

import asyncio
from unittest import mock

from mcp_server.splice_client.client import SpliceGRPCClient
from mcp_server.splice_client.models import PlanKind, SpliceCredits


def _fake_search_response():
    """A protobuf-shaped stand-in for a genuinely-empty SearchSamples reply."""
    response = mock.MagicMock()
    response.Samples = []
    response.TotalHits = 0
    response.MatchingTags = {}
    return response


class _StubbedClient(SpliceGRPCClient):
    """A client with a fully faked stub/pb2 — no real gRPC/TLS setup.

    `connect()` is overridden to simulate a cheap, always-available
    reconnect (as it would be against a locally-listening Splice
    process) while tracking how many times it's invoked.
    """

    def __init__(self, connect_succeeds: bool = True):
        self.channel = None
        self.stub = mock.MagicMock()
        self.connected = True
        self._port = None
        self._grpc = mock.MagicMock()
        self._pb2 = mock.MagicMock()
        self._pb2_grpc = mock.MagicMock()
        self._quota = mock.MagicMock()
        self._cached_credits = None
        self.degraded = False
        self.last_error = None
        self.connect_calls = 0
        self._connect_succeeds = connect_succeeds

    async def connect(self) -> bool:
        self.connect_calls += 1
        if self._connect_succeeds:
            self.connected = True
            self.degraded = False
            self.last_error = None
            return True
        self.connected = False
        return False


# ── 1. RPC failure marks degraded + resets connected ──────────────────


def test_rpc_failure_marks_degraded_and_resets_connected():
    client = _StubbedClient()
    client.stub.SearchSamples = mock.AsyncMock(side_effect=RuntimeError("splice down"))

    result = asyncio.run(client.search_samples("kick"))

    assert result.samples == []
    assert client.degraded is True
    assert client.connected is False
    assert client.last_error is not None
    assert "splice down" in client.last_error


def test_get_credits_failure_also_marks_degraded():
    client = _StubbedClient()
    client.stub.ValidateLogin = mock.AsyncMock(side_effect=RuntimeError("timeout"))

    result = asyncio.run(client.get_credits())

    assert result == SpliceCredits()
    assert client.degraded is True
    assert client.connected is False


def test_get_pack_info_failure_marks_degraded_and_keeps_error_message():
    client = _StubbedClient()
    client.stub.ListSamplePacks = mock.AsyncMock(side_effect=RuntimeError("channel closed"))

    pack, err = asyncio.run(client.get_pack_info("some-uuid"))

    assert pack is None
    assert err is not None and "channel closed" in err
    assert client.degraded is True
    assert client.connected is False


# ── 2. Next RPC after degradation retries connect() exactly once ──────


def test_next_rpc_after_degradation_reconnects_once_then_succeeds():
    client = _StubbedClient(connect_succeeds=True)
    client.stub.SearchSamples = mock.AsyncMock(side_effect=RuntimeError("boom"))

    # First call fails -> degraded. No reconnect attempted mid-call.
    asyncio.run(client.search_samples("kick"))
    assert client.degraded is True
    assert client.connected is False
    assert client.connect_calls == 0

    # Second call: _ensure_connected() should retry connect() exactly
    # once, then proceed with the (now working) RPC.
    client.stub.SearchSamples = mock.AsyncMock(return_value=_fake_search_response())
    result = asyncio.run(client.search_samples("kick"))

    assert client.connect_calls == 1
    assert result.samples == []
    assert client.degraded is False
    assert client.connected is True


def test_reconnect_attempt_that_fails_short_circuits_without_calling_rpc():
    """If the reconnect itself fails, the wrapper must not even attempt
    the RPC — it returns the default immediately, same as the original
    'not connected' guard."""
    client = _StubbedClient(connect_succeeds=False)
    client.stub.SearchSamples = mock.AsyncMock(side_effect=RuntimeError("boom"))
    asyncio.run(client.search_samples("kick"))
    assert client.degraded is True

    client.stub.SearchSamples = mock.AsyncMock(return_value=_fake_search_response())
    result = asyncio.run(client.search_samples("kick"))

    assert client.connect_calls == 1
    client.stub.SearchSamples.assert_not_called()
    assert result.samples == []
    # Still degraded — reconnect failed, so the unhealthy state persists.
    assert client.degraded is True
    assert client.connected is False


def test_ensure_connected_skips_reconnect_when_never_connected():
    """A client that was never successfully connected (and thus never
    marked degraded) must not attempt a reconnect from within a wrapper
    — this matches the historical `if not self.connected: return
    default` behavior and avoids touching `self.available`/`connect()`
    internals on partially constructed test doubles."""
    client = SpliceGRPCClient.__new__(SpliceGRPCClient)
    client.connected = False  # `degraded` attribute intentionally absent

    ok = asyncio.run(client._ensure_connected())

    assert ok is False


def test_client_degrades_gracefully_without_grpc_still_works():
    """Regression guard for the pre-existing bypass-__init__ pattern used
    elsewhere (test_splice_client.py) — must not crash now that guards
    call `_ensure_connected()` instead of reading `self.connected`
    directly."""
    c = SpliceGRPCClient.__new__(SpliceGRPCClient)
    c.connected = False

    assert asyncio.run(c.get_credits()) == SpliceCredits()
    result = asyncio.run(c.search_samples("anything"))
    assert result.samples == []
    assert asyncio.run(c.get_sample_info("abc")) is None


# ── 3. Successful connect() resets prior degradation ───────────────────


def test_successful_connect_clears_degraded_state():
    client = _StubbedClient()
    client._mark_degraded(RuntimeError("was down"))
    assert client.degraded is True
    assert client.connected is False

    ok = asyncio.run(client.connect())

    assert ok is True
    assert client.degraded is False
    assert client.last_error is None
    assert client.connected is True


# ── 4. degraded flag distinguishes failure from genuine empty result ──


def test_degraded_flag_distinguishes_genuine_empty_from_rpc_failure():
    # Genuine empty: the RPC succeeds but the catalog has zero matches.
    healthy_client = _StubbedClient()
    healthy_client.stub.SearchSamples = mock.AsyncMock(
        return_value=_fake_search_response()
    )
    healthy_result = asyncio.run(healthy_client.search_samples("no such thing"))
    assert healthy_result.samples == []
    assert healthy_client.degraded is False

    # Failure: the RPC raises. Same empty shape, but flagged unhealthy.
    unhealthy_client = _StubbedClient()
    unhealthy_client.stub.SearchSamples = mock.AsyncMock(
        side_effect=RuntimeError("splice desktop unreachable")
    )
    unhealthy_result = asyncio.run(unhealthy_client.search_samples("kick"))
    assert unhealthy_result.samples == []
    assert unhealthy_client.degraded is True
    assert unhealthy_client.last_error is not None

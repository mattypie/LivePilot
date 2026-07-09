"""Tests for SpliceHTTPBridge._request's retry/backoff loop.

Prior bugs:
  1. The retry loop slept unconditionally after EVERY attempt, including
     the final one — wasting a backoff delay right before re-raising and
     returning control to the caller.
  2. DECODE_ERROR (and NETWORK_ERROR) carry `status_code=0`, which is
     falsy and so slipped past the `exc.status_code and exc.status_code
     < 500` terminal check. A response-decode failure is deterministic —
     the same malformed bytes will fail to parse identically on every
     retry — so retrying it just burns the backoff delay for nothing.
"""

from __future__ import annotations

import asyncio
from unittest import mock

import pytest

from mcp_server.splice_client.http_bridge import (
    SpliceHTTPBridge,
    SpliceHTTPConfig,
    SpliceHTTPError,
)


def _bridge(max_retries: int = 2) -> SpliceHTTPBridge:
    cfg = SpliceHTTPConfig(max_retries=max_retries)
    return SpliceHTTPBridge(config=cfg, grpc_client=None)


def _patched_token():
    return mock.patch(
        "mcp_server.splice_client.http_bridge.fetch_session_token",
        new=mock.AsyncMock(return_value="fake-token"),
    )


def _patched_sleep():
    return mock.patch(
        "mcp_server.splice_client.http_bridge.asyncio.sleep",
        new=mock.AsyncMock(),
    )


# ── No sleep after the final attempt ───────────────────────────────────


def test_no_sleep_after_final_attempt():
    bridge = _bridge(max_retries=2)  # 3 total attempts
    call_count = {"n": 0}

    def _always_network_error(*_a, **_kw):
        call_count["n"] += 1
        raise SpliceHTTPError(
            code="NETWORK_ERROR", message="down", endpoint="/graphql",
            status_code=0,
        )

    with _patched_token(), mock.patch.object(
        bridge, "_perform_sync_request", side_effect=_always_network_error,
    ), _patched_sleep() as sleep_mock:
        with pytest.raises(SpliceHTTPError) as exc_info:
            asyncio.run(bridge._request("POST", "/graphql", body={"x": 1}))

    assert exc_info.value.code == "NETWORK_ERROR"
    # 1 + max_retries = 3 attempts total.
    assert call_count["n"] == 3
    # Sleeps happen BETWEEN attempts only: after attempt 1 and attempt 2,
    # never after the 3rd (final) attempt.
    assert sleep_mock.call_count == 2


def test_network_error_still_retries_and_can_succeed():
    """Sanity check that the terminal-error fix didn't also kill retries
    for genuinely transient errors."""
    bridge = _bridge(max_retries=2)
    call_count = {"n": 0}

    def _flaky(*_a, **_kw):
        call_count["n"] += 1
        if call_count["n"] < 2:
            raise SpliceHTTPError(
                code="NETWORK_ERROR", message="down", endpoint="/graphql",
                status_code=0,
            )
        return {"ok": True}

    with _patched_token(), mock.patch.object(
        bridge, "_perform_sync_request", side_effect=_flaky,
    ), _patched_sleep() as sleep_mock:
        result = asyncio.run(bridge._request("POST", "/graphql", body={"x": 1}))

    assert result == {"ok": True}
    assert call_count["n"] == 2
    assert sleep_mock.call_count == 1


# ── DECODE_ERROR is terminal — no retry at all ─────────────────────────


def test_decode_error_is_terminal_no_retry_no_sleep():
    bridge = _bridge(max_retries=2)
    call_count = {"n": 0}

    def _always_decode_error(*_a, **_kw):
        call_count["n"] += 1
        raise SpliceHTTPError(
            code="DECODE_ERROR", message="bad json", endpoint="/graphql",
        )

    with _patched_token(), mock.patch.object(
        bridge, "_perform_sync_request", side_effect=_always_decode_error,
    ), _patched_sleep() as sleep_mock:
        with pytest.raises(SpliceHTTPError) as exc_info:
            asyncio.run(bridge._request("POST", "/graphql", body={"x": 1}))

    assert exc_info.value.code == "DECODE_ERROR"
    assert call_count["n"] == 1  # no retry attempted
    sleep_mock.assert_not_called()


# ── 4xx (client errors) remain terminal — regression guard ─────────────


def test_client_error_status_remains_terminal_no_retry():
    bridge = _bridge(max_retries=2)
    call_count = {"n": 0}

    def _always_400(*_a, **_kw):
        call_count["n"] += 1
        raise SpliceHTTPError(
            code="HTTP_ERROR", message="bad request", endpoint="/graphql",
            status_code=400,
        )

    with _patched_token(), mock.patch.object(
        bridge, "_perform_sync_request", side_effect=_always_400,
    ), _patched_sleep() as sleep_mock:
        with pytest.raises(SpliceHTTPError) as exc_info:
            asyncio.run(bridge._request("POST", "/graphql", body={"x": 1}))

    assert exc_info.value.code == "HTTP_ERROR"
    assert call_count["n"] == 1
    sleep_mock.assert_not_called()


def test_no_retries_configured_means_single_attempt_no_sleep():
    bridge = _bridge(max_retries=0)
    call_count = {"n": 0}

    def _always_network_error(*_a, **_kw):
        call_count["n"] += 1
        raise SpliceHTTPError(
            code="NETWORK_ERROR", message="down", endpoint="/graphql",
            status_code=0,
        )

    with _patched_token(), mock.patch.object(
        bridge, "_perform_sync_request", side_effect=_always_network_error,
    ), _patched_sleep() as sleep_mock:
        with pytest.raises(SpliceHTTPError):
            asyncio.run(bridge._request("POST", "/graphql", body={"x": 1}))

    assert call_count["n"] == 1
    sleep_mock.assert_not_called()

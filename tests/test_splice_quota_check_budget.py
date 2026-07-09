"""Tests for DailyQuotaTracker.check_budget and its wiring into
SpliceGRPCClient.decide_download.

Prior bug: `decide_download`'s Ableton-plan branch gated on
`quota.summary()["at_limit"]` while `would_exceed()` / `near_limit()`
sat unused (dead code) despite being written for exactly this decision.
`check_budget()` computes all three predicates from a SINGLE
lock-protected read so a caller that needs more than one of them (as
`decide_download` does) can't observe two different moments if a
concurrent `record_download()` lands in between.
"""

from __future__ import annotations

import asyncio
import os
from unittest import mock

from mcp_server.splice_client.client import SpliceGRPCClient
from mcp_server.splice_client.models import PlanKind, SpliceCredits
from mcp_server.splice_client.quota import (
    DailyQuotaTracker,
    DEFAULT_DAILY_LIMIT,
    DEFAULT_WARN_THRESHOLD,
)


def _fresh_tracker(tmp_path, limit=DEFAULT_DAILY_LIMIT, warn=DEFAULT_WARN_THRESHOLD):
    return DailyQuotaTracker(
        path=os.path.join(tmp_path, "quota.json"),
        daily_limit=limit,
        warn_threshold=warn,
    )


# ── check_budget() shape + predicates ──────────────────────────────────


def test_check_budget_under_limit(tmp_path):
    t = _fresh_tracker(tmp_path, limit=5, warn=4)
    budget = t.check_budget(additional=1)
    assert budget["would_exceed"] is False
    assert budget["near_limit"] is False
    assert budget["at_limit"] is False
    assert budget["used_today"] == 0
    assert budget["remaining_today"] == 5
    assert budget["daily_limit"] == 5


def test_check_budget_would_exceed_at_boundary(tmp_path):
    t = _fresh_tracker(tmp_path, limit=5, warn=4)
    for _ in range(5):
        t.record_download("h")
    budget = t.check_budget(additional=1)
    assert budget["would_exceed"] is True
    assert budget["at_limit"] is True


def test_check_budget_near_limit_flag_without_exceeding(tmp_path):
    t = _fresh_tracker(tmp_path, limit=10, warn=8)
    for _ in range(8):
        t.record_download("h")
    budget = t.check_budget(additional=1)
    assert budget["near_limit"] is True
    assert budget["would_exceed"] is False  # 8 + 1 = 9 <= 10


def test_check_budget_respects_additional_count(tmp_path):
    t = _fresh_tracker(tmp_path, limit=10, warn=8)
    for _ in range(8):
        t.record_download("h")
    assert t.check_budget(additional=1)["would_exceed"] is False
    assert t.check_budget(additional=3)["would_exceed"] is True  # 8 + 3 = 11 > 10


def test_check_budget_agrees_with_standalone_predicates(tmp_path):
    """check_budget's fields must agree with the would_exceed/near_limit
    predicates it replaces — it's a combined atomic read, not new logic."""
    t = _fresh_tracker(tmp_path, limit=20, warn=15)
    for _ in range(17):
        t.record_download("h")
    budget = t.check_budget(additional=1)
    assert budget["would_exceed"] == t.would_exceed(additional=1)
    assert budget["near_limit"] == t.near_limit()
    assert budget["at_limit"] == (t.current()[0] >= t.daily_limit)


def test_check_budget_single_lock_acquisition(tmp_path):
    """Guards against a regression back to two separate current() calls
    (would_exceed() + near_limit()), which could observe different
    on-disk snapshots under concurrent writers."""
    t = _fresh_tracker(tmp_path, limit=10, warn=8)
    t.record_download("h")
    load_calls = {"n": 0}
    original_load = t._load

    def _counting_load():
        load_calls["n"] += 1
        return original_load()

    with mock.patch.object(t, "_load", side_effect=_counting_load):
        t.check_budget(additional=1)
    assert load_calls["n"] == 1


# ── Wired into decide_download (not dead code) ─────────────────────────


class _FakeAbletonClient(SpliceGRPCClient):
    def __init__(self, tmp_path, daily_limit=100, warn_threshold=90):
        tracker = DailyQuotaTracker(
            path=os.path.join(tmp_path, "quota.json"),
            daily_limit=daily_limit,
            warn_threshold=warn_threshold,
        )
        super().__init__(quota_tracker=tracker)
        self.connected = True

    async def get_credits(self) -> SpliceCredits:
        return SpliceCredits(
            credits=80, plan="ableton_live", plan_kind=PlanKind.ABLETON_LIVE,
        )


def test_decide_download_routes_through_check_budget_not_summary(tmp_path):
    client = _FakeAbletonClient(tmp_path)
    with mock.patch.object(
        client._quota, "check_budget", wraps=client._quota.check_budget,
    ) as spy_check, mock.patch.object(
        client._quota, "summary", wraps=client._quota.summary,
    ) as spy_summary:
        decision = asyncio.run(client.decide_download("h"))

    spy_check.assert_called_once_with(additional=1)
    spy_summary.assert_not_called()
    assert decision.allowed


def test_decide_download_surfaces_near_limit_warning_in_reason(tmp_path):
    client = _FakeAbletonClient(tmp_path, daily_limit=100, warn_threshold=90)
    for i in range(90):
        client._quota.record_download(f"h-{i}")

    decision = asyncio.run(client.decide_download("h"))

    assert decision.allowed
    assert "approaching" in decision.reason.lower()


def test_decide_download_blocks_at_would_exceed_boundary(tmp_path):
    client = _FakeAbletonClient(tmp_path, daily_limit=100, warn_threshold=90)
    for i in range(100):
        client._quota.record_download(f"h-{i}")

    decision = asyncio.run(client.decide_download("h"))

    assert not decision.allowed
    assert decision.gating_mode == "daily_quota"
    assert decision.quota_used == 100

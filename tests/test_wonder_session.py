"""Unit tests for WonderSession and WonderDiagnosis models."""

from mcp_server.wonder_mode.session import (
    WonderDiagnosis,
    WonderSession,
    get_wonder_session,
    store_wonder_session,
    _wonder_sessions,
)


def setup_function():
    _wonder_sessions.clear()


# ── Creation and storage ─────────────────────────────────────────


def test_session_creation():
    ws = WonderSession(session_id="ws_001", request_text="make it magical")
    assert ws.session_id == "ws_001"
    assert ws.status == "diagnosing"
    assert ws.outcome == "pending"
    assert ws.variant_count_actual == 0
    assert ws.variants == []


def test_store_and_retrieve():
    ws = WonderSession(session_id="ws_002", request_text="test")
    store_wonder_session(ws)
    retrieved = get_wonder_session("ws_002")
    assert retrieved is ws


def test_retrieve_missing_returns_none():
    assert get_wonder_session("nonexistent") is None


def test_eviction_at_capacity():
    for i in range(12):
        store_wonder_session(
            WonderSession(session_id=f"ws_{i:03d}", request_text=f"req {i}")
        )
    # First 2 should be evicted (max 10)
    assert get_wonder_session("ws_000") is None
    assert get_wonder_session("ws_001") is None
    # Last 10 should remain
    assert get_wonder_session("ws_002") is not None
    assert get_wonder_session("ws_011") is not None


# ── Status transitions ───────────────────────────────────────────


def test_status_defaults_to_diagnosing():
    ws = WonderSession(session_id="ws_s", request_text="test")
    assert ws.status == "diagnosing"


def test_valid_transitions():
    ws = WonderSession(session_id="ws_t", request_text="test")
    assert ws.transition_to("variants_ready") is True
    assert ws.status == "variants_ready"
    assert ws.transition_to("previewing") is True
    assert ws.status == "previewing"
    assert ws.transition_to("resolved") is True
    assert ws.status == "resolved"


def test_invalid_transitions_rejected():
    ws = WonderSession(session_id="ws_inv", request_text="test")
    # Can't go from diagnosing to resolved directly
    assert ws.transition_to("resolved") is False
    assert ws.status == "diagnosing"
    # Can't go from diagnosing to previewing
    assert ws.transition_to("previewing") is False
    assert ws.status == "diagnosing"


def test_resolved_is_terminal():
    ws = WonderSession(session_id="ws_term", request_text="test")
    ws.transition_to("variants_ready")
    ws.transition_to("resolved")
    # Can't transition from resolved
    assert ws.transition_to("diagnosing") is False
    assert ws.transition_to("variants_ready") is False
    assert ws.status == "resolved"


# ── Degradation ──────────────────────────────────────────────────


def test_degraded_reason_set():
    ws = WonderSession(
        session_id="ws_d",
        request_text="test",
        variant_count_actual=1,
        degraded_reason="Only 1 distinct executable move found",
    )
    assert ws.degraded_reason != ""
    assert ws.variant_count_actual == 1


# ── WonderDiagnosis ──────────────────────────────────────────────


def test_diagnosis_creation():
    diag = WonderDiagnosis(
        trigger_reason="user_request",
        problem_class="exploration",
        current_identity="Dark minimal techno",
        sacred_elements=[{"element_type": "groove", "description": "808 kick"}],
        blocked_dimensions=[],
        candidate_domains=[],
    )
    assert diag.trigger_reason == "user_request"
    assert diag.problem_class == "exploration"
    assert diag.confidence == 0.0
    assert diag.variant_budget == 3
    assert diag.degraded_capabilities == []


def test_diagnosis_to_dict():
    diag = WonderDiagnosis(
        trigger_reason="stuckness_detected",
        problem_class="overpolished_loop",
        current_identity="Ambient drone",
        sacred_elements=[],
        blocked_dimensions=["energy"],
        candidate_domains=["arrangement", "transition"],
        confidence=0.7,
        degraded_capabilities=["song_brain"],
    )
    d = diag.to_dict()
    assert d["trigger_reason"] == "stuckness_detected"
    assert d["problem_class"] == "overpolished_loop"
    assert d["candidate_domains"] == ["arrangement", "transition"]
    assert d["confidence"] == 0.7
    assert "song_brain" in d["degraded_capabilities"]


# ── State-layer hardening: lock + session fingerprint ────────────


def test_session_fingerprint_defaults_empty():
    """Absent fingerprint means 'no signal' — must default to empty string
    so older/degraded sessions still commit without a staleness check."""
    ws = WonderSession(session_id="ws_fp_default", request_text="test")
    assert ws.session_fingerprint == ""


def test_session_fingerprint_stamped_and_carried_in_to_dict():
    ws = WonderSession(
        session_id="ws_fp",
        request_text="test",
        session_fingerprint="abc123",
    )
    assert ws.session_fingerprint == "abc123"
    assert ws.to_dict()["session_fingerprint"] == "abc123"


def test_store_wonder_session_concurrent_hammer_does_not_raise():
    """Regression: _wonder_sessions is mutated from both threadpooled sync
    tools (enter_wonder_mode) and event-loop async tools (commit_preview_variant
    via find_session_by_preview_set). Before the fix, store_wonder_session's
    check-then-evict loop raced the same way as preview_studio's — two
    threads could grab the same oldest_key and a second pop/del raised.

    sys.setswitchinterval is lowered modestly (10 microseconds — 500x more
    aggressive than the 5ms default, but well short of the 1us/0.1us extremes
    that make thread scheduling itself pathologically expensive on a loaded
    machine) to force fine-grained interleaving without risking a
    multi-minute stall under CI/CPU contention. Verified in isolation to
    reproduce the pre-fix KeyError/RuntimeError reliably (dozens of times
    per run) while completing in well under a second.
    """
    import sys
    import threading
    from mcp_server.wonder_mode.session import _wonder_sessions_lock

    old_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-5)
    with _wonder_sessions_lock:
        _wonder_sessions.clear()
    errors: list[Exception] = []

    def _worker(worker_id: int) -> None:
        try:
            for j in range(150):
                store_wonder_session(
                    WonderSession(
                        session_id=f"race_{worker_id}_{j}",
                        request_text="race",
                    )
                )
        except Exception as exc:  # pragma: no cover - failure path only
            errors.append(exc)

    try:
        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        sys.setswitchinterval(old_interval)

    assert errors == [], f"concurrent store_wonder_session raised: {errors!r}"
    with _wonder_sessions_lock:
        size = len(_wonder_sessions)
    assert size <= 10  # _MAX_WONDER_SESSIONS

"""WonderSession and WonderDiagnosis — thin lifecycle coordinator.

WonderSession ties the Wonder lifecycle together: diagnosis, variant
generation, preview, commit/discard, and outcome recording.

WonderDiagnosis is a structured diagnosis built from stuckness,
SongBrain, action ledger, and creative threads.
"""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field
from typing import Optional


_MAX_WONDER_SESSIONS = 10


@dataclass
class WonderDiagnosis:
    """Structured diagnosis driving Wonder variant generation."""

    trigger_reason: str  # "user_request", "stuckness_detected", "repeated_undos"
    problem_class: str  # from RESCUE_TYPES + "exploration"
    current_identity: str  # from SongBrain.identity_core
    sacred_elements: list[dict] = field(default_factory=list)
    blocked_dimensions: list[str] = field(default_factory=list)
    candidate_domains: list[str] = field(default_factory=list)
    variant_budget: int = 3
    confidence: float = 0.0
    degraded_capabilities: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class WonderSession:
    """Thin lifecycle coordinator for a Wonder Mode session."""

    session_id: str
    request_text: str
    kernel_id: str = ""

    # Diagnosis
    diagnosis: Optional[WonderDiagnosis] = None

    # Lifecycle references
    creative_thread_id: str = ""
    preview_set_id: str = ""

    # Variants
    variants: list[dict] = field(default_factory=list)
    recommended: str = ""
    variant_count_actual: int = 0

    # Outcome
    selected_variant_id: str = ""
    outcome: str = "pending"  # pending, committed, rejected_all, abandoned

    # Degradation
    degraded_reason: str = ""

    status: str = "diagnosing"  # diagnosing, variants_ready, previewing, resolved

    # Session-identity guard — see preview_studio.models.compute_session_fingerprint.
    # Stamped at enter_wonder_mode time from the session_info already fetched
    # for diagnosis; carried onto any PreviewSet built from this session's
    # variants so a later commit can detect a stale topology. Empty string
    # means "no signal" (older sessions, or built without a live session).
    session_fingerprint: str = ""

    # Valid state transitions
    _VALID_TRANSITIONS: dict = field(default_factory=lambda: {
        "diagnosing": {"variants_ready"},
        "variants_ready": {"previewing", "resolved"},
        "previewing": {"resolved"},
        "resolved": set(),  # terminal
    }, repr=False)

    def transition_to(self, new_status: str) -> bool:
        """Attempt a state transition. Returns False if invalid."""
        valid = self._VALID_TRANSITIONS.get(self.status, set())
        if new_status not in valid:
            return False
        self.status = new_status
        return True

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.diagnosis:
            d["diagnosis"] = self.diagnosis.to_dict()
        return d


# ── In-memory store ───────────────────────────────────────────────
#
# _wonder_sessions is mutated from both threadpooled sync tools
# (enter_wonder_mode) and event-loop async tools (commit_preview_variant,
# via _find_wonder_session_by_preview) — without a lock, two concurrent
# store_wonder_session calls can race in the check-then-evict loop below:
# both read the same `oldest_key`, both pop it, and a naive `del` on the
# second call raises KeyError. One module-level lock around the
# get/store/find critical sections closes that race.

_wonder_sessions: dict[str, WonderSession] = {}
_wonder_sessions_lock = threading.Lock()


def store_wonder_session(ws: WonderSession) -> None:
    """Store a WonderSession with FIFO eviction at capacity."""
    with _wonder_sessions_lock:
        _wonder_sessions[ws.session_id] = ws
        while len(_wonder_sessions) > _MAX_WONDER_SESSIONS:
            oldest_key = next(iter(_wonder_sessions), None)
            if oldest_key is None:
                break
            evicted = _wonder_sessions.pop(oldest_key, None)
            if evicted is not None and evicted.outcome == "pending":
                evicted.outcome = "abandoned"


def get_wonder_session(session_id: str) -> Optional[WonderSession]:
    """Retrieve a WonderSession by ID."""
    with _wonder_sessions_lock:
        return _wonder_sessions.get(session_id)


def find_session_by_preview_set(set_id: str) -> Optional[WonderSession]:
    """Find a WonderSession linked to a preview set ID."""
    with _wonder_sessions_lock:
        for ws in _wonder_sessions.values():
            if ws.preview_set_id == set_id:
                return ws
    return None

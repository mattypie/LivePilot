"""Preview Studio data models — pure dataclasses, zero I/O."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

from ..runtime.degradation import DegradationInfo


def compute_session_fingerprint(session_info: Optional[dict]) -> str:
    """Lightweight fingerprint of session topology (track_count + ordered
    track names).

    PreviewSet and WonderSession cache compiled plans that reference
    positional track/device indices. Those indices are only valid against
    the session topology that existed when the plan was compiled — if the
    user adds/removes/reorders tracks before the plan is committed, replaying
    those indices can silently hit the wrong track.

    This helper is the single source of truth for that fingerprint, shared
    by preview_studio and wonder_mode so both stamp/verify it the same way.
    Callers stamp it at creation time from session_info they already fetched
    (never triggers its own Ableton round-trip). At commit/replay time they
    fetch a fresh session_info and compare fingerprints.

    Returns "" when session_info is missing/malformed — an empty fingerprint
    means "no signal available" and callers must skip the staleness check
    (this keeps older cached/persisted objects, built before this field
    existed, committable).
    """
    if not isinstance(session_info, dict) or not session_info:
        return ""
    tracks = session_info.get("tracks")
    if not isinstance(tracks, list):
        tracks = []
    names = [str(t.get("name", "")) for t in tracks if isinstance(t, dict)]
    track_count = session_info.get("track_count", len(names))
    seed = json.dumps({"track_count": track_count, "track_names": names}, sort_keys=True)
    return hashlib.sha256(seed.encode()).hexdigest()[:16]


@dataclass
class PreviewVariant:
    """One creative option in a preview set."""

    variant_id: str = ""
    label: str = ""  # "safe", "strong", "unexpected"
    intent: str = ""  # what this variant is trying to achieve
    novelty_level: float = 0.0  # 0=conservative, 1=radical
    songbrain_delta: str = ""  # what changed vs identity
    taste_fit: float = 0.5  # 0-1 how well it matches user taste
    render_ref: str = ""  # reference to cached render
    summary: str = ""  # one-line musical explanation

    # What changed, why it matters, what it preserves
    what_changed: str = ""
    why_it_matters: str = ""
    what_preserved: str = ""

    # Move / plan data
    move_id: str = ""
    compiled_plan: Optional[dict] = None

    # Scoring
    score: float = 0.0
    identity_effect: str = "preserves"  # preserves, evolves, contrasts, resets

    # State
    status: str = "pending"  # pending, rendered, committed, discarded
    preview_mode: str = ""  # audible_preview, metadata_only_preview, analytical_preview
    created_at_ms: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        # Remove None compiled_plan for cleaner output
        if d.get("compiled_plan") is None:
            d.pop("compiled_plan", None)
        return d


@dataclass
class PreviewSet:
    """A set of variants tied to one user request."""

    set_id: str = ""
    request_text: str = ""
    strategy: str = "creative_triptych"  # creative_triptych, binary, custom
    source_kernel_id: str = ""
    variants: list[PreviewVariant] = field(default_factory=list)
    comparison: Optional[dict] = None
    committed_variant_id: str = ""
    status: str = "pending"  # pending, compared, committed, discarded
    created_at_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    # Degradation signalling — set when the engine substituted a fallback
    # (e.g. an empty-but-valid kernel) during variant compilation. Callers
    # can inspect .degradation.is_degraded to tell synthesized preview
    # topology apart from a real kernel-backed compile.
    degradation: DegradationInfo = field(default_factory=DegradationInfo)
    # Session-identity guard — see compute_session_fingerprint(). Stamped at
    # creation time from the session_info the creator already fetched. Empty
    # string means "no signal" (older objects, or created without a live
    # session) and callers must skip the staleness check.
    session_fingerprint: str = ""

    def to_dict(self) -> dict:
        return {
            "set_id": self.set_id,
            "request_text": self.request_text,
            "strategy": self.strategy,
            "source_kernel_id": self.source_kernel_id,
            "variants": [v.to_dict() for v in self.variants],
            "comparison": self.comparison,
            "committed_variant_id": self.committed_variant_id,
            "status": self.status,
            "variant_count": len(self.variants),
            "degradation": self.degradation.to_dict(),
            "session_fingerprint": self.session_fingerprint,
        }

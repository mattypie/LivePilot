"""Snapshot Normalizer — canonical input normalization for all evaluators.

Ensures analyzer outputs are in a consistent schema regardless of
which tool produced them. All evaluators should consume normalized
snapshots, never raw tool outputs.

Design: AGENT_OS_PHASE0_HARDENING_PLAN.md, section 3.2
"""

from __future__ import annotations

import time
from typing import Optional


_RICH_ANALYZER_KEYS = (
    "spectral_shape",
    "mel_bands",
    "chroma",
    "onset",
    "onsets",
    "novelty",
    "loudness",
    "sub_detail",
)


def normalize_sonic_snapshot(
    raw: Optional[dict],
    source: str = "unknown",
) -> Optional[dict]:
    """Normalize a raw analyzer/perception output into canonical snapshot form.

    Accepts both {"bands": {...}} and {"spectrum": {...}} shapes.
    Rich analyzer streams are preserved when present so evaluators can
    reason from FluCoMa character descriptors instead of collapsing every
    decision down to the 9-band spectrum.
    Returns None if input is empty or None.

    Canonical form:
    {
        "spectrum": {band: value, ...},
        "rms": float or None,
        "peak": float or None,
        "detected_key": str or None,
        "spectral_shape": dict or None,
        "mel_bands": list or None,
        "chroma": dict/list or None,
        "onset": dict or None,
        "novelty": dict or None,
        "loudness": dict or None,
        "source": str,
        "normalized_at_ms": int,
    }
    """
    if not raw or not isinstance(raw, dict):
        return None

    bands = raw.get("spectrum") or raw.get("bands") or {}
    has_rich_analyzer_data = any(raw.get(k) is not None for k in _RICH_ANALYZER_KEYS)
    if not bands and not has_rich_analyzer_data:
        return None

    normalized = {
        "spectrum": bands,
        "rms": raw.get("rms"),
        "peak": raw.get("peak"),
        "detected_key": raw.get("key") or raw.get("detected_key"),
        "source": source,
        "normalized_at_ms": int(time.time() * 1000),
    }

    for key in _RICH_ANALYZER_KEYS:
        if key in raw and raw.get(key) is not None:
            out_key = "onset" if key == "onsets" else key
            normalized[out_key] = raw.get(key)

    return normalized

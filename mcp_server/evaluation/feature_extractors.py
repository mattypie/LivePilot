"""Feature Extractors — derive measurable values from normalized snapshots.

Replicates the dimension-extraction logic from _agent_os_engine but operates
on the canonical normalized snapshot format (always has "spectrum" key).

All returned values are clamped to 0.0-1.0 for consistent scoring.
"""

from __future__ import annotations

import math
from typing import Any, Optional

from ..tools._evaluation_contracts import MEASURABLE_DIMENSIONS


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp value to [lo, hi] range."""
    return max(lo, min(hi, value))


def _number(value: Any) -> Optional[float]:
    """Best-effort numeric coercion for analyzer payloads."""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _nested_number(payload: Any, *keys: str) -> Optional[float]:
    """Read a numeric value from a dict payload using candidate keys."""
    if isinstance(payload, dict):
        for key in keys:
            value = _number(payload.get(key))
            if value is not None:
                return value
    return _number(payload)


def _centroid_to_unit(centroid: float) -> float:
    """Map centroid-like values to 0..1.

    FluCoMa deployments may report centroid in Hz or normalized units.
    Values <= 1 are treated as normalized. Larger values are mapped across
    a practical musical range where 150 Hz is very dark and 8 kHz is bright.
    """
    if centroid <= 1.0:
        return _clamp(centroid)
    return _clamp((centroid - 150.0) / (8000.0 - 150.0))


def _lufs_to_unit(lufs: float) -> float:
    """Map momentary LUFS to a rough 0..1 energy proxy."""
    return _clamp((lufs + 60.0) / 60.0)


def extract_dimension_value(
    snapshot: dict,
    dimension: str,
) -> Optional[float]:
    """Extract a measurable dimension value from a normalized sonic snapshot.

    Args:
        snapshot: Normalized snapshot (from normalize_sonic_snapshot).
                  Must have a "spectrum" key with band values.
        dimension: One of the MEASURABLE_DIMENSIONS (brightness, warmth,
                   weight, clarity, density, energy, punch).

    Returns:
        Float in 0.0-1.0 for measurable dimensions, None otherwise.
    """
    if not snapshot or not isinstance(snapshot, dict):
        return None

    bands = snapshot.get("spectrum") or {}
    spectral_shape = snapshot.get("spectral_shape") or {}
    onset = snapshot.get("onset") or {}
    novelty = snapshot.get("novelty") or {}
    loudness = snapshot.get("loudness") or {}

    rms = snapshot.get("rms")
    peak = snapshot.get("peak")

    if dimension == "brightness":
        centroid = _nested_number(spectral_shape, "centroid", "centroid_hz")
        if centroid is not None:
            return _centroid_to_unit(centroid)
        if not bands:
            return None
        high = bands.get("high", 0)
        presence = bands.get("presence", 0)
        return _clamp((high + presence) / 2.0)

    elif dimension == "warmth":
        if not bands:
            return None
        return _clamp(bands.get("low_mid", 0))

    elif dimension == "weight":
        if not bands:
            return None
        sub = bands.get("sub_low", bands.get("sub", 0))
        low = bands.get("low", 0)
        return _clamp((sub + low) / 2.0)

    elif dimension == "clarity":
        if not bands:
            return None
        low_mid = bands.get("low_mid", 0)
        return _clamp(1.0 - low_mid)

    elif dimension == "density":
        flatness = _nested_number(spectral_shape, "flatness", "spectral_flatness")
        if flatness is not None:
            return _clamp(flatness)
        if not bands:
            return None
        vals = [max(v, 1e-10) for v in bands.values()
                if isinstance(v, (int, float))]
        if not vals:
            return None
        geo_mean = math.exp(sum(math.log(v) for v in vals) / len(vals))
        arith_mean = sum(vals) / len(vals)
        return _clamp(geo_mean / max(arith_mean, 1e-10))

    elif dimension == "energy":
        rms_value = _number(rms)
        if rms_value is not None:
            return _clamp(rms_value)
        lufs = _nested_number(loudness, "momentary_lufs", "lufs", "integrated_lufs")
        if lufs is not None:
            return _lufs_to_unit(lufs)
        return None

    elif dimension == "punch":
        rms_value = _number(rms)
        peak_value = _number(peak)
        if rms_value and peak_value and rms_value > 0:
            crest_db = 20.0 * math.log10(max(peak_value / rms_value, 1.0))
            return _clamp(crest_db / 20.0)
        onset_strength = _nested_number(onset, "strength", "onset")
        if onset_strength is not None:
            return _clamp(onset_strength)
        spectral_crest = _nested_number(spectral_shape, "crest")
        if spectral_crest is not None:
            return _clamp(spectral_crest)
        return None

    elif dimension == "novelty":
        novelty_score = _nested_number(novelty, "score", "novelty", "value")
        return _clamp(novelty_score) if novelty_score is not None else None

    elif dimension == "motion":
        novelty_score = _nested_number(novelty, "score", "novelty", "value")
        onset_strength = _nested_number(onset, "strength", "onset")
        vals = [v for v in (novelty_score, onset_strength) if v is not None]
        if vals:
            return _clamp(sum(vals) / len(vals))
        return None

    else:
        # Unmeasurable dimension
        return None


def _label_low_mid_high(value: float, low: str, mid: str, high: str) -> str:
    if value < 0.33:
        return low
    if value > 0.67:
        return high
    return mid


def extract_character_profile(snapshot: dict) -> dict:
    """Summarize analyzer data as a production-oriented character profile.

    This is intentionally descriptive, not prescriptive. Engines can attach
    the profile to their analysis response so the agent chooses sound-source,
    device, and parameter moves before reaching for generic level changes.
    """
    if not snapshot or not isinstance(snapshot, dict):
        return {"available": False, "values": {}, "labels": {}, "biases": []}

    dimensions = (
        "brightness", "warmth", "weight", "clarity", "density",
        "energy", "punch", "motion", "novelty",
    )
    values = {
        dim: round(val, 4)
        for dim in dimensions
        if (val := extract_dimension_value(snapshot, dim)) is not None
    }

    labels: dict[str, str] = {}
    if "brightness" in values:
        labels["brightness"] = _label_low_mid_high(values["brightness"], "dark", "balanced", "bright")
    if "warmth" in values:
        labels["warmth"] = _label_low_mid_high(values["warmth"], "lean", "warm", "thick")
    if "weight" in values:
        labels["weight"] = _label_low_mid_high(values["weight"], "light", "grounded", "heavy")
    if "density" in values:
        labels["density"] = _label_low_mid_high(values["density"], "peaked", "shaped", "flat/noisy")
    if "punch" in values:
        labels["punch"] = _label_low_mid_high(values["punch"], "soft", "defined", "spiky")
    if "motion" in values:
        labels["motion"] = _label_low_mid_high(values["motion"], "static", "moving", "busy")

    biases: list[str] = []
    if values.get("brightness", 0.5) > 0.72:
        biases.append("prefer filter tone, source choice, or de-harshing over lowering track volume")
    if values.get("brightness", 0.5) < 0.28:
        biases.append("prefer oscillator/filter opening, excitation, or air-band source choice over level boosts")
    if values.get("motion", 0.5) < 0.25:
        biases.append("prefer modulation, envelope drift, or evolving devices before static mix moves")
    if values.get("punch", 0.5) < 0.25:
        biases.append("prefer envelope/transient shaping or source layering before pushing volume")
    if values.get("density", 0.0) > 0.75:
        biases.append("prefer subtractive filtering or simpler source selection when the spectrum is flat/noisy")
    if values.get("weight", 0.5) < 0.25:
        biases.append("prefer instrument/register/source changes for low-end weight before master gain")

    return {
        "available": bool(values),
        "values": values,
        "labels": labels,
        "biases": biases,
    }

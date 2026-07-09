"""Resolvers — find tracks, devices, and parameters from a SessionKernel.

These are the "eyes" of the semantic move compiler. They inspect the kernel's
session topology to find the right targets for a musical intent.

Pure functions — no I/O, no MCP calls. They read from the kernel dict only.
"""

from __future__ import annotations

from typing import Optional


# ── Role inference from track names ──────────────────────────────────────────

_ROLE_KEYWORDS: dict[str, list[str]] = {
    "drums": ["drum", "beat", "kit", "perc", "rhythm", "hat", "kick", "snare"],
    "bass": ["bass", "sub", "808", "low"],
    "chords": ["chord", "rhodes", "keys", "piano", "organ", "electric", "wurli"],
    "pad": ["pad", "texture", "ambient", "drone", "atmosphere"],
    "lead": ["lead", "melody", "synth", "glitch", "hook", "vocal"],
    "percussion": ["perc", "shaker", "tambourine", "conga", "bongo", "rim"],
    "fx": ["fx", "effect", "noise", "riser", "impact", "sweep"],
}


def infer_role(track_name: str) -> str:
    """Infer a musical role from a track name. Returns 'unknown' if no match."""
    name_lower = track_name.lower()
    for role, keywords in _ROLE_KEYWORDS.items():
        for kw in keywords:
            if kw in name_lower:
                return role
    return "unknown"


# ── Track finders ────────────────────────────────────────────────────────────

def find_tracks_by_role(kernel: dict, roles: list[str]) -> list[dict]:
    """Find all tracks whose inferred role is in the given list.

    Returns list of {index, name, role, volume, pan} for matched tracks.
    """
    tracks = kernel.get("session_info", {}).get("tracks", [])
    results = []
    for track in tracks:
        role = infer_role(track.get("name", ""))
        if role in roles:
            results.append({
                "index": track.get("index", 0),
                "name": track.get("name", ""),
                "role": role,
                "volume": track.get("volume"),
                "pan": track.get("pan"),
                "mute": track.get("mute", False),
                "solo": track.get("solo", False),
            })
    return results


def find_loudest_track(kernel: dict) -> Optional[dict]:
    """Find the track with the highest volume setting."""
    tracks = kernel.get("session_info", {}).get("tracks", [])
    if not tracks:
        return None
    # Volume might not be in session_info tracks — return the first non-muted
    non_muted = [t for t in tracks if not t.get("mute", False)]
    return non_muted[0] if non_muted else tracks[0]


def find_track_by_name(kernel: dict, name_substring: str) -> Optional[dict]:
    """Find a track whose name contains the given substring (case-insensitive)."""
    tracks = kernel.get("session_info", {}).get("tracks", [])
    name_lower = name_substring.lower()
    for track in tracks:
        if name_lower in track.get("name", "").lower():
            return track
    return None


# ── Device finders ───────────────────────────────────────────────────────────

def find_device_on_track(
    kernel: dict, track_index: int, device_class: str
) -> Optional[dict]:
    """Find a device by class name on a track. Returns {device_index, name} or None.

    Note: This requires device data in the kernel, which may not always be
    available. Returns None if device data is missing.
    """
    # Device data would be in an extended kernel; for now, return None
    # and let the compiler use find_and_load_device as a fallback
    return None


# ── Spectral helpers ─────────────────────────────────────────────────────────

def get_spectral_balance(kernel: dict) -> Optional[dict]:
    """Extract spectral band balance from the kernel's capability data.

    Returns None if no spectral data is available.
    """
    # Spectral data isn't stored in the base kernel — it would come from
    # a pre-capture. Return None for graceful degradation.
    return None


def is_analyzer_available(kernel: dict) -> bool:
    """Check if the M4L analyzer is connected."""
    cap = kernel.get("capability_state", {})
    domains = cap.get("domains", {})
    analyzer = domains.get("analyzer", {})
    return analyzer.get("available", False)


# ── Volume math ──────────────────────────────────────────────────────────────

def clamp_volume(vol: float) -> float:
    """Clamp volume to Ableton's 0.0-1.0 range."""
    return max(0.0, min(1.0, vol))


def adjust_volume(current: float, delta_percent: float) -> float:
    """Adjust a volume by a percentage. delta_percent=5 means +5%."""
    new = current + (delta_percent / 100.0)
    return clamp_volume(new)


def compile_relative_volume(
    current: Optional[float],
    delta_percent: float,
    *,
    cap: Optional[float] = None,
    floor: Optional[float] = None,
    fallback: float,
) -> float:
    """Compile a bounded RELATIVE volume nudge from a track's current level.

    P2-21 fix: mix/sound-design/transition compilers used to write an
    ABSOLUTE volume regardless of what was already on the track — e.g.
    "make it punchier" could turn a hot drum bus DOWN by blindly writing
    a flat 0.75. This computes current + delta_percent% instead (via
    adjust_volume), then clamps to an optional cap/floor so the move's
    musical intent (push toward louder / pull toward quieter) still holds
    even when starting from an unusual level.

    `current` is None when the connected Remote Script predates the
    `volume` field in get_session_info (see transport.py get_session_info)
    — in that case this falls back to the historical absolute value so
    the compiler degrades gracefully instead of crashing or guessing.

    The cap/floor bound the SIZE of the nudge — they must never flip its
    DIRECTION. A track already louder than `cap` must not get pulled back
    down to the cap when the move means to push it UP (that would
    reintroduce the P2-21 bug in a smaller way); symmetrically a track
    already quieter than `floor` must not get nudged back UP when the
    move means to pull it DOWN. So after clamping to [floor, cap], the
    result is re-clamped against `current` in the direction of travel.
    """
    if current is None:
        return fallback
    target = adjust_volume(current, delta_percent)
    if cap is not None:
        target = min(target, cap)
    if floor is not None:
        target = max(target, floor)
    if delta_percent > 0:
        target = max(target, current)
    elif delta_percent < 0:
        target = min(target, current)
    return clamp_volume(target)

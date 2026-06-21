"""MCP-side Live version capabilities model.

Pure data model — no I/O. Parses version info from get_session_info
responses and exposes feature flags for tool routing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class LiveVersionCapabilities:
    """Feature availability based on detected Live version."""

    major: int = 12
    minor: int = 0
    patch: int = 0

    @classmethod
    def from_version_string(cls, version_str: str) -> LiveVersionCapabilities:
        """Parse '12.3.6' into a capabilities instance.

        Tolerant of malformed input: extracts the leading integer of each
        dotted component and falls back to the conservative floor
        (major=12, minor=0, patch=0) for missing or non-numeric components.
        Mirrors the Remote Script's degrade-to-(12,0,0) contract
        (remote_script/LivePilot/version_detect.py) so a junk live_version
        string never crashes get_capability_state / get_session_kernel /
        --doctor.
        """
        def _component(parts: list[str], index: int, default: int) -> int:
            if index >= len(parts):
                return default
            match = re.match(r"\s*(\d+)", parts[index])
            return int(match.group(1)) if match else default

        parts = (version_str or "").split(".")
        major = _component(parts, 0, 12)
        minor = _component(parts, 1, 0)
        patch = _component(parts, 2, 0)
        return cls(major=major, minor=minor, patch=patch)

    @classmethod
    def from_session_info(cls, session_info: dict) -> LiveVersionCapabilities:
        """Extract version from get_session_info response.

        Looks for 'live_version' field. Falls back to 12.0.0 if absent
        (pre-upgrade Remote Script).
        """
        version_str = session_info.get("live_version", "12.0.0")
        return cls.from_version_string(version_str)

    @property
    def _version_tuple(self) -> tuple[int, int, int]:
        return (self.major, self.minor, self.patch)

    # ── Feature flags ──────────────────────────────────────────────

    @property
    def has_native_arrangement_clips(self) -> bool:
        """Track.create_midi_clip(start, length) — 12.1.10+"""
        return self._version_tuple >= (12, 1, 10)

    @property
    def has_display_value(self) -> bool:
        """DeviceParameter.display_value — 12.2+"""
        return self._version_tuple >= (12, 2, 0)

    @property
    def has_insert_device(self) -> bool:
        """Track.insert_device(name, index?) — 12.3+"""
        return self._version_tuple >= (12, 3, 0)

    @property
    def has_drum_rack_construction(self) -> bool:
        """insert_chain + DrumChain.in_note — 12.3+"""
        return self._version_tuple >= (12, 3, 0)

    @property
    def has_take_lanes(self) -> bool:
        """Take Lanes API — 12.2+"""
        return self._version_tuple >= (12, 2, 0)

    @property
    def has_stem_separation(self) -> bool:
        """Stem separation via MFL — 12.3+"""
        return self._version_tuple >= (12, 3, 0)

    @property
    def has_link_audio(self) -> bool:
        """Live 12.4 Link Audio UX exists; MCP control remains probe-gated."""
        return self._version_tuple >= (12, 4, 0)

    @property
    def has_stem_time_selection(self) -> bool:
        """Live 12.4 selected-time stem separation UX exists; probe before use."""
        return self._version_tuple >= (12, 4, 0)

    @property
    def has_stem_merge_selected(self) -> bool:
        """Live 12.4 merge-selected-stems UX exists; probe before use."""
        return self._version_tuple >= (12, 4, 0)

    @property
    def has_replace_sample_native(self) -> bool:
        """SimplerDevice.replace_sample(path) — 12.4+"""
        return self._version_tuple >= (12, 4, 0)

    @property
    def capability_tier(self) -> str:
        """Human-readable tier: core | enhanced_arrangement | full_intelligence | collaborative."""
        if self._version_tuple >= (12, 4, 0):
            return "collaborative"
        elif self._version_tuple >= (12, 3, 0):
            return "full_intelligence"
        elif self._version_tuple >= (12, 1, 10):
            return "enhanced_arrangement"
        else:
            return "core"

    def to_dict(self) -> dict:
        """Serialize for API responses and capability probes."""
        return {
            "version": f"{self.major}.{self.minor}.{self.patch}",
            "capability_tier": self.capability_tier,
            "native_arrangement_clips": self.has_native_arrangement_clips,
            "display_value": self.has_display_value,
            "insert_device": self.has_insert_device,
            "drum_rack_construction": self.has_drum_rack_construction,
            "take_lanes": self.has_take_lanes,
            "stem_separation": self.has_stem_separation,
            "link_audio": self.has_link_audio,
            "stem_time_selection": self.has_stem_time_selection,
            "stem_merge_selected": self.has_stem_merge_selected,
            "replace_sample_native": self.has_replace_sample_native,
        }

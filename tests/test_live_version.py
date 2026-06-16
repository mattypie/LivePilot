"""Tests for MCP-side Live version capabilities model."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mcp_server.runtime.live_version import LiveVersionCapabilities


def test_parse_version_string():
    caps = LiveVersionCapabilities.from_version_string("12.3.6")
    assert caps.major == 12
    assert caps.minor == 3
    assert caps.patch == 6


def test_parse_version_string_two_parts():
    caps = LiveVersionCapabilities.from_version_string("12.3")
    assert caps.patch == 0


def test_feature_detection_12_0():
    caps = LiveVersionCapabilities(12, 0, 0)
    assert not caps.has_native_arrangement_clips
    assert not caps.has_display_value
    assert not caps.has_insert_device
    assert not caps.has_drum_rack_construction


def test_feature_detection_12_1_10():
    caps = LiveVersionCapabilities(12, 1, 10)
    assert caps.has_native_arrangement_clips
    assert not caps.has_display_value
    assert not caps.has_insert_device


def test_feature_detection_12_2():
    caps = LiveVersionCapabilities(12, 2, 0)
    assert caps.has_native_arrangement_clips
    assert caps.has_display_value
    assert not caps.has_insert_device


def test_feature_detection_12_3():
    caps = LiveVersionCapabilities(12, 3, 0)
    assert caps.has_native_arrangement_clips
    assert caps.has_display_value
    assert caps.has_insert_device
    assert caps.has_drum_rack_construction


def test_to_dict():
    caps = LiveVersionCapabilities(12, 3, 6)
    d = caps.to_dict()
    assert d["version"] == "12.3.6"
    assert d["native_arrangement_clips"] is True
    assert d["insert_device"] is True


def test_capability_tier_12_0():
    caps = LiveVersionCapabilities(12, 0, 0)
    assert caps.capability_tier == "core"


def test_capability_tier_12_1_10():
    caps = LiveVersionCapabilities(12, 1, 10)
    assert caps.capability_tier == "enhanced_arrangement"


def test_capability_tier_12_3():
    caps = LiveVersionCapabilities(12, 3, 0)
    assert caps.capability_tier == "full_intelligence"


# ── Live 12.4+ (Collaborative tier) ──────────────────────────────────

def test_capability_tier_12_4_0():
    caps = LiveVersionCapabilities(12, 4, 0)
    assert caps.capability_tier == "collaborative"


def test_capability_tier_12_4_11():
    caps = LiveVersionCapabilities(12, 4, 11)
    assert caps.capability_tier == "collaborative"


def test_capability_tier_12_5_forward_compat():
    """Future versions stay in Collaborative until a new tier is defined."""
    caps = LiveVersionCapabilities(12, 5, 0)
    assert caps.capability_tier == "collaborative"


def test_has_replace_sample_native_12_4_0():
    caps = LiveVersionCapabilities(12, 4, 0)
    assert caps.has_replace_sample_native is True


def test_has_replace_sample_native_12_3_6():
    caps = LiveVersionCapabilities(12, 3, 6)
    assert caps.has_replace_sample_native is False


def test_live_12_4_flags_link_audio_and_selected_time_stems_as_version_known():
    """12.4 introduces Link Audio and selected-time stem UX, but runtime
    support is still probe-gated elsewhere."""
    caps = LiveVersionCapabilities(12, 4, 2)
    assert caps.has_link_audio is True
    assert caps.has_stem_time_selection is True
    assert caps.has_stem_merge_selected is True


def test_live_12_3_does_not_claim_12_4_link_or_selected_time_stem_features():
    caps = LiveVersionCapabilities(12, 3, 6)
    assert caps.has_link_audio is False
    assert caps.has_stem_time_selection is False
    assert caps.has_stem_merge_selected is False


def test_to_dict_12_4_includes_collaborative_and_native():
    caps = LiveVersionCapabilities(12, 4, 0)
    d = caps.to_dict()
    assert d["capability_tier"] == "collaborative"
    assert d["replace_sample_native"] is True
    assert d["link_audio"] is True
    assert d["stem_time_selection"] is True
    assert d["stem_merge_selected"] is True

"""Tests for Reference Engine V1 — profiles, gap analysis, tactic routing.

~20 tests covering models, profile builders, gap analyzer, tactic router.
All pure-function tests, no I/O.
"""

from __future__ import annotations

import math

import pytest

from mcp_server.reference_engine.models import (
    GapEntry,
    GapReport,
    ReferencePlan,
    ReferenceProfile,
)
from mcp_server.reference_engine.profile_builder import (
    build_audio_reference_profile,
    build_style_reference_profile,
)
from mcp_server.reference_engine.gap_analyzer import (
    analyze_gaps,
    classify_gap_relevance,
    detect_identity_warnings,
)
from mcp_server.reference_engine.tactic_router import (
    build_reference_plan,
    route_to_engines,
)


# ── Model tests ────────────────────────────────────────────────────


class TestReferenceProfile:
    def test_default_source_type(self):
        p = ReferenceProfile()
        assert p.source_type == "audio"

    def test_to_dict_roundtrip(self):
        p = ReferenceProfile(
            source_type="style",
            loudness_posture=-14.0,
            spectral_contour={"centroid": 2000},
            harmonic_character="warm_harmonic",
            transition_tendencies=["reveal", "drift"],
        )
        d = p.to_dict()
        assert d["source_type"] == "style"
        assert d["loudness_posture"] == -14.0
        assert d["harmonic_character"] == "warm_harmonic"
        assert len(d["transition_tendencies"]) == 2

    def test_to_dict_has_all_fields(self):
        d = ReferenceProfile().to_dict()
        expected_keys = {
            "source_type", "loudness_posture", "spectral_contour",
            "width_depth", "density_arc", "section_pacing",
            "harmonic_character", "transition_tendencies",
        }
        assert set(d.keys()) == expected_keys


class TestGapEntry:
    def test_to_dict(self):
        g = GapEntry(domain="spectral", delta=0.05, relevant=True)
        d = g.to_dict()
        assert d["domain"] == "spectral"
        assert d["delta"] == 0.05

    def test_defaults(self):
        g = GapEntry()
        assert g.relevant is True
        assert g.identity_warning is False


class TestGapReport:
    def test_relevant_gaps_property(self):
        report = GapReport(
            reference_id="test",
            gaps=[
                GapEntry(domain="loudness", delta=2.0, relevant=True),
                GapEntry(domain="width", delta=0.01, relevant=False),
                GapEntry(domain="spectral", delta=0.05, relevant=True),
            ],
        )
        assert len(report.relevant_gaps) == 2
        assert all(g.relevant for g in report.relevant_gaps)

    def test_identity_warnings_property(self):
        report = GapReport(
            reference_id="test",
            gaps=[
                GapEntry(domain="harmonic", delta=1.0, identity_warning=True),
                GapEntry(domain="loudness", delta=2.0, identity_warning=False),
            ],
        )
        warnings = report.identity_warnings
        assert len(warnings) == 1
        assert "harmonic" in warnings[0]

    def test_to_dict_includes_computed_properties(self):
        report = GapReport(
            reference_id="test_ref",
            gaps=[GapEntry(domain="loudness", delta=3.0, relevant=True)],
            overall_distance=3.0,
        )
        d = report.to_dict()
        assert "relevant_gaps" in d
        assert "identity_warnings" in d
        assert d["overall_distance"] == 3.0
        assert len(d["relevant_gaps"]) == 1


class TestReferencePlan:
    def test_to_dict(self):
        plan = ReferencePlan(
            gap_report=GapReport(reference_id="r1"),
            ranked_tactics=[{"rank": 1, "tactic": "EQ cut"}],
            target_engines=["mix_engine"],
        )
        d = plan.to_dict()
        assert d["target_engines"] == ["mix_engine"]
        assert len(d["ranked_tactics"]) == 1
        assert d["gap_report"]["reference_id"] == "r1"


# ── Profile builder tests ─────────────────────────────────────────


class TestBuildAudioProfile:
    def test_basic_audio_profile(self):
        comparison = {
            "reference_lufs": -14.0,
            "centroid_delta_hz": 200.0,
            "stereo_width_ref": 0.25,
            "band_deltas": {"sub_60hz": 0.01, "mid_2khz": -0.03},
        }
        p = build_audio_reference_profile(comparison)
        assert p.source_type == "audio"
        assert p.loudness_posture == -14.0
        assert p.width_depth["stereo_width"] == 0.25
        assert "band_balance" in p.spectral_contour

    def test_missing_fields_default_gracefully(self):
        p = build_audio_reference_profile({})
        assert p.source_type == "audio"
        assert p.loudness_posture == 0.0
        assert p.spectral_contour["band_balance"] == {}


class TestBuildStyleProfile:
    def test_empty_tactics(self):
        p = build_style_reference_profile([])
        assert p.source_type == "style"
        assert p.transition_tendencies == []

    def test_single_tactic(self):
        tactic = {
            "artist_or_genre": "techno",
            "tactic_name": "rolling_groove",
            "arrangement_patterns": ["long_intro_16bars", "minimal_variation"],
            "device_chain": [{"name": "Compressor"}, {"name": "Delay"}],
            "automation_gestures": ["drift", "release"],
        }
        p = build_style_reference_profile([tactic])
        assert p.source_type == "style"
        assert "drift" in p.transition_tendencies
        assert "release" in p.transition_tendencies
        assert len(p.section_pacing) == 2
        assert len(p.density_arc) == 1

    def test_harmonic_character_inference(self):
        tactic = {
            "artist_or_genre": "ambient",
            "tactic_name": "texture_bed",
            "arrangement_patterns": [],
            "device_chain": [
                {"name": "Reverb"},
                {"name": "Auto Filter"},
            ],
            "automation_gestures": [],
        }
        p = build_style_reference_profile([tactic])
        assert p.harmonic_character == "atmospheric_filtered"

    def test_loud_chain_maps_loudness_posture_to_lufs_not_score(self):
        """A limiter/glue-heavy chain must yield loudness_posture as integrated
        LUFS (the documented contract), NOT a raw [-1,+1] posture score. The
        [-1,+1] heuristic is mapped onto the LUFS axis (+1 → -8 LUFS)."""
        tactic = {
            "artist_or_genre": "techno",
            "tactic_name": "loud_master",
            "arrangement_patterns": [],
            "device_chain": [
                {"name": "Glue Compressor", "params": {}},
                {"name": "Limiter", "params": {}},
            ],
            "automation_gestures": [],
        }
        p = build_style_reference_profile([tactic])
        # Posture +1 (loud) → -14 + 1*6 = -8.0 LUFS, not the raw score 1.0.
        assert p.loudness_posture == pytest.approx(-8.0, abs=0.01)

    def test_reverb_heavy_chain_maps_to_quieter_lufs(self):
        """A reverb-heavy (dynamic/quiet) chain → posture -1 → -20 LUFS."""
        tactic = {
            "artist_or_genre": "ambient",
            "tactic_name": "wash",
            "arrangement_patterns": [],
            "device_chain": [{"name": "Reverb", "params": {"Dry/Wet": 0.7}}],
            "automation_gestures": [],
        }
        p = build_style_reference_profile([tactic])
        assert p.loudness_posture == pytest.approx(-20.0, abs=0.01)

    def test_no_loudness_signal_neutral_lufs_no_spurious_gap(self):
        """Regression: a chain with no loudness-signaling devices must map to
        the -14 LUFS neutral, NOT 0.0. Against a typical -14 LUFS project this
        yields ~no loudness gap — whereas the old raw 0.0 read as a bogus
        ~14 LU gap in analyze_reference_gaps."""
        tactic = {
            "artist_or_genre": "lofi",
            "tactic_name": "neutral",
            "arrangement_patterns": [],
            "device_chain": [{"name": "EQ Eight", "params": {}}],
            "automation_gestures": [],
        }
        p = build_style_reference_profile([tactic])
        assert p.loudness_posture == pytest.approx(-14.0, abs=0.01)

        report = analyze_gaps({"loudness": -14.0, "loudness_unit": "lufs"}, p)
        loudness_gaps = [g for g in report.gaps if g.domain == "loudness"]
        assert loudness_gaps, "loudness gap entry should exist"
        # The whole point: delta ~0 and NOT flagged relevant (old code: -14.0).
        assert loudness_gaps[0].delta == pytest.approx(0.0, abs=0.01)
        assert loudness_gaps[0].relevant is False


# ── Gap analyzer tests ─────────────────────────────────────────────


class TestAnalyzeGaps:
    def test_loudness_gap(self):
        snapshot = {"loudness": -10.0}
        ref = ReferenceProfile(loudness_posture=-14.0)
        report = analyze_gaps(snapshot, ref)
        loudness_gaps = [g for g in report.gaps if g.domain == "loudness"]
        assert len(loudness_gaps) == 1
        assert loudness_gaps[0].delta == pytest.approx(4.0, abs=0.01)

    def test_spectral_gap(self):
        snapshot = {"spectral": {"band_balance": {"mid_2khz": 0.3}}}
        ref = ReferenceProfile(
            spectral_contour={"band_balance": {"mid_2khz": 0.2}},
        )
        report = analyze_gaps(snapshot, ref)
        spectral_gaps = [g for g in report.gaps if g.domain == "spectral"]
        assert len(spectral_gaps) == 1
        assert spectral_gaps[0].delta == pytest.approx(0.1, abs=0.001)

    def test_width_gap(self):
        snapshot = {"width": 0.4}
        ref = ReferenceProfile(width_depth={"stereo_width": 0.2})
        report = analyze_gaps(snapshot, ref)
        width_gaps = [g for g in report.gaps if g.domain == "width"]
        assert len(width_gaps) == 1
        assert width_gaps[0].delta == pytest.approx(0.2, abs=0.001)

    def test_harmonic_gap_identity_warning(self):
        snapshot = {"harmonic_character": "diatonic_major"}
        ref = ReferenceProfile(harmonic_character="minor_modal")
        report = analyze_gaps(snapshot, ref)
        harmonic_gaps = [g for g in report.gaps if g.domain == "harmonic"]
        assert len(harmonic_gaps) == 1
        assert harmonic_gaps[0].identity_warning is True

    def test_empty_snapshot_and_ref(self):
        report = analyze_gaps({}, ReferenceProfile())
        assert report.overall_distance == 0.0

    def test_overall_distance_positive(self):
        snapshot = {"loudness": -8.0, "width": 0.5}
        ref = ReferenceProfile(
            loudness_posture=-14.0,
            width_depth={"stereo_width": 0.1},
        )
        report = analyze_gaps(snapshot, ref)
        assert report.overall_distance > 0


class TestClassifyGapRelevance:
    def test_with_goal_dimensions(self):
        gap = GapEntry(domain="spectral", delta=0.05, relevant=True)
        assert classify_gap_relevance(gap, ["spectral", "width"]) is True
        assert classify_gap_relevance(gap, ["loudness"]) is False

    def test_empty_goal_keeps_original(self):
        gap = GapEntry(domain="spectral", delta=0.05, relevant=False)
        assert classify_gap_relevance(gap, []) is False


class TestDetectIdentityWarnings:
    def test_warns_on_identity_gaps(self):
        gaps = [
            GapEntry(domain="harmonic", delta=1.0, identity_warning=True),
            GapEntry(domain="loudness", delta=3.0, identity_warning=False),
        ]
        warnings = detect_identity_warnings(gaps)
        assert len(warnings) == 1
        assert "harmonic" in warnings[0]

    def test_no_warnings(self):
        gaps = [GapEntry(domain="loudness", delta=3.0, identity_warning=False)]
        assert detect_identity_warnings(gaps) == []


# ── Tactic router tests ───────────────────────────────────────────


class TestRouteToEngines:
    def test_spectral_routes_to_mix(self):
        report = GapReport(
            reference_id="test",
            gaps=[GapEntry(domain="spectral", delta=0.05, relevant=True)],
        )
        routes = route_to_engines(report)
        assert len(routes) == 1
        assert routes[0]["engine"] == "mix_engine"

    def test_pacing_routes_to_composition(self):
        report = GapReport(
            reference_id="test",
            gaps=[GapEntry(domain="pacing", delta=3.0, relevant=True)],
        )
        routes = route_to_engines(report)
        assert routes[0]["engine"] == "composition"

    def test_irrelevant_gaps_excluded(self):
        report = GapReport(
            reference_id="test",
            gaps=[
                GapEntry(domain="spectral", delta=0.05, relevant=False),
                GapEntry(domain="loudness", delta=3.0, relevant=True),
            ],
        )
        routes = route_to_engines(report)
        assert len(routes) == 1
        assert routes[0]["domain"] == "loudness"

    def test_sorted_by_priority(self):
        report = GapReport(
            reference_id="test",
            gaps=[
                GapEntry(domain="loudness", delta=1.5, relevant=True),
                GapEntry(domain="spectral", delta=0.1, relevant=True),
            ],
        )
        routes = route_to_engines(report)
        assert routes[0]["priority"] >= routes[1]["priority"]


class TestBuildReferencePlan:
    def test_plan_has_all_components(self):
        report = GapReport(
            reference_id="test",
            gaps=[
                GapEntry(domain="spectral", delta=0.05, relevant=True,
                         suggested_tactic="EQ cut"),
                GapEntry(domain="pacing", delta=2.0, relevant=True,
                         suggested_tactic="Add sections"),
            ],
        )
        plan = build_reference_plan(report)
        assert len(plan.ranked_tactics) == 2
        assert "mix_engine" in plan.target_engines
        assert "composition" in plan.target_engines

    def test_plan_to_dict(self):
        report = GapReport(reference_id="r1")
        plan = build_reference_plan(report)
        d = plan.to_dict()
        assert "gap_report" in d
        assert "ranked_tactics" in d
        assert "target_engines" in d

    def test_identity_warned_gaps_deprioritized(self):
        report = GapReport(
            reference_id="test",
            gaps=[
                GapEntry(domain="harmonic", delta=1.0, relevant=True,
                         identity_warning=True, suggested_tactic="Reharmonize"),
                GapEntry(domain="loudness", delta=8.0, relevant=True,
                         identity_warning=False, suggested_tactic="Gain up"),
            ],
        )
        plan = build_reference_plan(report)
        # Loudness should rank higher despite harmonic having delta=1.0
        assert plan.ranked_tactics[0]["domain"] == "loudness"


# ── _fetch_project_snapshot integration tests (P1: spectral key + width) ──


class _FakeSpectral:
    """Minimal SpectralCache stand-in for _fetch_project_snapshot."""

    def __init__(self, values):
        self._values = values
        self.is_connected = True

    def get(self, key):
        return self._values.get(key)


class _FakeAbleton:
    def __init__(self, session_info):
        self._session_info = session_info

    def send_command(self, command, params):
        if command == "get_session_info":
            return self._session_info
        return {}


class _FakeCtx:
    def __init__(self, lifespan_context):
        self.lifespan_context = lifespan_context


def _make_ctx(spectrum=None, tracks=None):
    spectral_values = {}
    if spectrum is not None:
        spectral_values["spectrum"] = {"value": spectrum}
        spectral_values["rms"] = {"value": 0.1}
    session_info = {
        "track_count": len(tracks or []),
        "scene_count": 2,
        "tracks": tracks or [],
    }
    return _FakeCtx({
        "ableton": _FakeAbleton(session_info),
        "spectral": _FakeSpectral(spectral_values),
    })


def test_fetch_snapshot_spectrum_under_band_balance():
    """The project spectrum must land under 'band_balance' so it lines up
    with gap_analyzer / the reference profile — not the phantom 'bands' key."""
    from mcp_server.reference_engine.tools import _fetch_project_snapshot

    proj_bands = {"sub": 0.45, "low": 0.5, "mid": 0.6, "high": 0.3}
    ctx = _make_ctx(spectrum=proj_bands)
    snapshot = _fetch_project_snapshot(ctx)

    assert "band_balance" in snapshot["spectral"]
    assert snapshot["spectral"]["band_balance"] == proj_bands
    # The old phantom key must be gone.
    assert "bands" not in snapshot["spectral"]


def test_snapshot_spectrum_feeds_gap_analyzer():
    """End-to-end: a populated project spectrum produces a real (non-phantom)
    spectral gap rather than a delta equal to the full reference band."""
    from mcp_server.reference_engine.tools import _fetch_project_snapshot

    ctx = _make_ctx(spectrum={"mid": 0.30})
    snapshot = _fetch_project_snapshot(ctx)
    ref = ReferenceProfile(spectral_contour={"band_balance": {"mid": 0.20}})
    report = analyze_gaps(snapshot, ref)
    spectral_gaps = [g for g in report.gaps if g.domain == "spectral"]
    assert len(spectral_gaps) == 1
    # 0.30 (project) - 0.20 (reference) = 0.10, NOT 0.0 - 0.20 = -0.20.
    assert spectral_gaps[0].delta == pytest.approx(0.10, abs=1e-6)


def test_fetch_snapshot_width_populated_from_pans():
    """Project stereo width must be estimated from track pans, not left 0.0."""
    from mcp_server.reference_engine.tools import _fetch_project_snapshot

    tracks = [
        {"name": "a", "mixer": {"panning": 0.8}},
        {"name": "b", "mixer": {"panning": -0.4}},
    ]
    ctx = _make_ctx(spectrum={"mid": 0.5}, tracks=tracks)
    snapshot = _fetch_project_snapshot(ctx)

    # mean(|0.8|, |-0.4|) = 0.6
    assert snapshot["width"] == pytest.approx(0.6, abs=1e-6)


def test_fetch_snapshot_width_zero_when_no_tracks():
    """No tracks -> width stays at the 0.0 default (no crash)."""
    from mcp_server.reference_engine.tools import _fetch_project_snapshot

    ctx = _make_ctx(spectrum={"mid": 0.5}, tracks=[])
    snapshot = _fetch_project_snapshot(ctx)
    assert snapshot["width"] == 0.0


# ── P2-35: audio profile stores absolute ref bands, not deltas ────────


class TestAudioProfileAbsoluteBands:
    def test_absolute_reference_bands_preferred(self):
        """P2-35: build_audio_reference_profile must store the reference's
        OWN absolute band_balance, never band_deltas (mix - ref)."""
        comparison = {
            "reference_lufs": -14.0,
            "band_deltas": {"sub_60hz": 0.05, "mid_2khz": -0.03},
            "reference_band_balance": {"sub_60hz": 0.40, "mid_2khz": 0.25},
            "mix_band_balance": {"sub_60hz": 0.45, "mid_2khz": 0.22},
        }
        p = build_audio_reference_profile(comparison)
        bands = p.spectral_contour["band_balance"]
        # Absolute reference levels, NOT the deltas.
        assert bands == {"sub_60hz": 0.40, "mid_2khz": 0.25}
        assert bands != comparison["band_deltas"]

    def test_reconstructs_ref_from_mix_minus_delta(self):
        """When only mix bands + deltas are present, reconstruct ref = mix - delta."""
        comparison = {
            "band_deltas": {"sub_60hz": 0.05, "mid_2khz": -0.03},
            "mix_band_balance": {"sub_60hz": 0.45, "mid_2khz": 0.22},
        }
        p = build_audio_reference_profile(comparison)
        bands = p.spectral_contour["band_balance"]
        assert bands["sub_60hz"] == pytest.approx(0.40, abs=1e-6)  # 0.45 - 0.05
        assert bands["mid_2khz"] == pytest.approx(0.25, abs=1e-6)  # 0.22 - (-0.03)

    def test_identical_mix_yields_zero_spectral_gap(self):
        """End-to-end P2-35: an identical mix (deltas all 0, ref == mix bands)
        produces ~zero spectral gaps — not a gap equal to the band energy."""
        mix_bands = {"sub_60hz": 0.45, "mid_2khz": 0.30}
        comparison = {
            "reference_lufs": -14.0,
            "band_deltas": {b: 0.0 for b in mix_bands},
            "reference_band_balance": dict(mix_bands),
            "mix_band_balance": dict(mix_bands),
        }
        profile = build_audio_reference_profile(comparison)
        snapshot = {"spectral": {"band_balance": dict(mix_bands)}}
        report = analyze_gaps(snapshot, profile)
        spectral_gaps = [g for g in report.gaps if g.domain == "spectral"]
        assert spectral_gaps == []  # no band crosses the 0.01 relevance threshold

    def test_old_delta_as_levels_behavior_is_gone(self):
        """Regression guard: the buggy path stored band_deltas as band_balance,
        so an identical mix produced a spurious gap == project band energy."""
        mix_bands = {"sub_60hz": 0.45, "mid_2khz": 0.30}
        comparison = {
            "band_deltas": {b: 0.0 for b in mix_bands},
            "reference_band_balance": dict(mix_bands),
            "mix_band_balance": dict(mix_bands),
        }
        profile = build_audio_reference_profile(comparison)
        # If the old bug were present, band_balance would equal the (zero)
        # deltas and the identical-mix gap would equal each band's energy.
        assert profile.spectral_contour["band_balance"] == mix_bands


# ── P2-36: loudness compared on a common scale (LUFS vs LUFS) ─────────


class TestLoudnessScaleMatch:
    def test_rms_dbfs_converted_to_lufs_before_delta(self):
        """P2-36: a project loudness tagged rms_dbfs must be converted onto
        the integrated-LUFS axis before subtracting loudness_posture."""
        # Reference at -14 LUFS. Project RMS dBFS of -14.0 is ~-14.691 LUFS,
        # so the gap should be ~-0.691 LU, NOT 0.0.
        snapshot = {"loudness": -14.0, "loudness_unit": "rms_dbfs"}
        ref = ReferenceProfile(loudness_posture=-14.0)
        report = analyze_gaps(snapshot, ref)
        loud = [g for g in report.gaps if g.domain == "loudness"][0]
        assert loud.delta == pytest.approx(-0.69, abs=0.01)

    def test_lufs_unit_passes_through_unchanged(self):
        """A project loudness already in LUFS is differenced directly."""
        snapshot = {"loudness": -10.0, "loudness_unit": "lufs"}
        ref = ReferenceProfile(loudness_posture=-14.0)
        report = analyze_gaps(snapshot, ref)
        loud = [g for g in report.gaps if g.domain == "loudness"][0]
        assert loud.delta == pytest.approx(4.0, abs=0.01)

    def test_identical_loudness_yields_near_zero_gap(self):
        """An identical project (same LUFS, both tagged) yields ~zero loudness gap."""
        snapshot = {"loudness": -14.0, "loudness_unit": "lufs"}
        ref = ReferenceProfile(loudness_posture=-14.0)
        report = analyze_gaps(snapshot, ref)
        loud = [g for g in report.gaps if g.domain == "loudness"][0]
        assert loud.delta == pytest.approx(0.0, abs=0.01)
        assert loud.relevant is False  # below the 1.0 LU relevance threshold

    def test_snapshot_default_unit_is_rms_dbfs(self):
        """The live snapshot derives RMS, so it must declare loudness_unit."""
        from mcp_server.reference_engine.tools import _fetch_project_snapshot

        ctx = _make_ctx(spectrum={"mid": 0.5})
        snapshot = _fetch_project_snapshot(ctx)
        assert snapshot["loudness_unit"] == "rms_dbfs"


# ── P2-37: overall_distance normalizes per-domain before summing ──────


class TestOverallDistanceNormalization:
    def test_pacing_does_not_dominate_spectral(self):
        """P2-37: a small (1-section) pacing delta must not overwhelm a large
        spectral delta after normalization."""
        # pacing delta = 1 section (scale 2.0 -> 0.5 normalized)
        # spectral delta = 0.1 in a band (scale 0.1 -> 1.0 normalized)
        snapshot = {
            "spectral": {"band_balance": {"mid": 0.30}},
            "pacing": [{"label": "a"}],
        }
        ref = ReferenceProfile(
            spectral_contour={"band_balance": {"mid": 0.20}},
            section_pacing=[],  # proj has 1, ref has 0 -> pacing delta 1
        )
        report = analyze_gaps(snapshot, ref)
        spectral = next(g for g in report.gaps if g.domain == "spectral")
        pacing = next(g for g in report.gaps if g.domain == "pacing")
        # Raw: pacing delta 1.0 >> spectral delta 0.1. After normalization the
        # spectral term (1.0) should exceed the pacing term (0.5).
        assert abs(pacing.delta) > abs(spectral.delta)  # raw magnitudes
        spectral_norm = abs(spectral.delta) / 0.1
        pacing_norm = abs(pacing.delta) / 2.0
        assert spectral_norm > pacing_norm

    def test_distance_uses_normalized_terms(self):
        """The reported overall_distance must equal the normalized Euclidean sum,
        not the raw-delta sum."""
        snapshot = {
            "spectral": {"band_balance": {"mid": 0.30}},
            "pacing": [{"label": "a"}],
        }
        ref = ReferenceProfile(
            spectral_contour={"band_balance": {"mid": 0.20}},
            section_pacing=[],
        )
        report = analyze_gaps(snapshot, ref)
        # Normalized: spectral (0.1/0.1)=1.0, pacing (1.0/2.0)=0.5
        expected = math.sqrt(1.0 ** 2 + 0.5 ** 2)
        assert report.overall_distance == pytest.approx(expected, abs=1e-3)
        # Raw (buggy) distance would be sqrt(0.1**2 + 1.0**2) ≈ 1.005.
        raw = math.sqrt(0.1 ** 2 + 1.0 ** 2)
        assert report.overall_distance != pytest.approx(raw, abs=1e-3)

    def test_identical_project_yields_zero_distance(self):
        """An identical project (zero deltas everywhere) -> ~zero distance."""
        bands = {"sub_60hz": 0.45, "mid_2khz": 0.30}
        snapshot = {
            "loudness": -14.0,
            "loudness_unit": "lufs",
            "spectral": {"band_balance": dict(bands)},
            "width": 0.2,
            "pacing": [],
        }
        ref = ReferenceProfile(
            loudness_posture=-14.0,
            spectral_contour={"band_balance": dict(bands)},
            width_depth={"stereo_width": 0.2},
            section_pacing=[],
        )
        report = analyze_gaps(snapshot, ref)
        assert report.overall_distance == pytest.approx(0.0, abs=1e-6)

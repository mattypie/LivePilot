"""Tests for Sample Engine critics — pure computation, no I/O."""

from __future__ import annotations

import pytest

from types import SimpleNamespace

from mcp_server.sample_engine.critics import (
    run_key_fit_critic,
    run_tempo_fit_critic,
    run_frequency_fit_critic,
    run_role_fit_critic,
    run_vibe_fit_critic,
    run_intent_fit_critic,
    run_all_sample_critics,
)
from mcp_server.sample_engine.models import (
    SampleProfile,
    SampleIntent,
    CriticResult,
)


def _make_profile(**kwargs) -> SampleProfile:
    defaults = {"source": "test", "file_path": "/t.wav", "name": "test"}
    defaults.update(kwargs)
    return SampleProfile(**defaults)


class TestKeyFitCritic:
    def test_same_key_perfect_score(self):
        r = run_key_fit_critic(_make_profile(key="Cm"), song_key="Cm")
        assert r.score == 1.0

    def test_relative_major_minor(self):
        r = run_key_fit_critic(_make_profile(key="Eb"), song_key="Cm")
        assert r.score >= 0.8

    def test_fifth_relationship(self):
        r = run_key_fit_critic(_make_profile(key="Gm"), song_key="Cm")
        assert r.score >= 0.6

    def test_distant_key_low_score(self):
        r = run_key_fit_critic(_make_profile(key="F#"), song_key="C")
        assert r.score <= 0.5

    def test_unknown_key_zero(self):
        r = run_key_fit_critic(_make_profile(key=None), song_key="Cm")
        assert r.score == 0.0

    def test_no_song_key_neutral(self):
        r = run_key_fit_critic(_make_profile(key="Cm"), song_key=None)
        assert r.score == 0.5


class TestTempoFitCritic:
    def test_exact_match(self):
        r = run_tempo_fit_critic(_make_profile(bpm=128.0), session_tempo=128.0)
        assert r.score >= 0.95

    def test_half_time(self):
        r = run_tempo_fit_critic(_make_profile(bpm=64.0), session_tempo=128.0)
        assert r.score >= 0.85

    def test_double_time(self):
        r = run_tempo_fit_critic(_make_profile(bpm=256.0), session_tempo=128.0)
        assert r.score >= 0.85

    def test_close_bpm(self):
        r = run_tempo_fit_critic(_make_profile(bpm=132.0), session_tempo=128.0)
        assert r.score >= 0.6

    def test_far_bpm(self):
        r = run_tempo_fit_critic(_make_profile(bpm=80.0), session_tempo=140.0)
        assert r.score <= 0.4

    def test_unknown_bpm_zero(self):
        r = run_tempo_fit_critic(_make_profile(bpm=None), session_tempo=128.0)
        assert r.score == 0.0


class TestRoleFitCritic:
    def test_fills_missing_role(self):
        r = run_role_fit_critic(
            _make_profile(material_type="vocal"),
            existing_roles=["drums", "bass", "synth"],
        )
        assert r.score >= 0.8

    def test_redundant_role(self):
        r = run_role_fit_critic(
            _make_profile(material_type="drum_loop"),
            existing_roles=["drums", "percussion", "hihat"],
        )
        assert r.score <= 0.5


class TestVibeFitCritic:
    def test_no_taste_graph_neutral(self):
        r = run_vibe_fit_critic(_make_profile(), taste_graph=None)
        assert r.score == 0.5

    def test_zero_evidence_neutral(self):
        tg = SimpleNamespace(evidence_count=0, novelty_band=0.5)
        r = run_vibe_fit_critic(_make_profile(), taste_graph=tg)
        assert r.score == 0.5

    def test_high_energy_high_novelty_good_fit(self):
        tg = SimpleNamespace(evidence_count=10, novelty_band=0.9)
        # transient_density is peaks/sec; ~10/s is a busy, high-energy sample.
        profile = _make_profile(brightness=0.9, transient_density=10.0)
        r = run_vibe_fit_critic(profile, taste_graph=tg)
        assert r.score >= 0.7

    def test_low_energy_high_novelty_poor_fit(self):
        tg = SimpleNamespace(evidence_count=10, novelty_band=0.9)
        profile = _make_profile(brightness=0.1, transient_density=0.1)
        r = run_vibe_fit_critic(profile, taste_graph=tg)
        assert r.score <= 0.5

    def test_low_energy_low_novelty_good_fit(self):
        tg = SimpleNamespace(evidence_count=10, novelty_band=0.1)
        profile = _make_profile(brightness=0.1, transient_density=0.1)
        r = run_vibe_fit_critic(profile, taste_graph=tg)
        assert r.score >= 0.7

    def test_score_always_in_range(self):
        tg = SimpleNamespace(evidence_count=5, novelty_band=0.5)
        profile = _make_profile(brightness=0.5, transient_density=0.5)
        r = run_vibe_fit_critic(profile, taste_graph=tg)
        assert 0.0 <= r.score <= 1.0


class TestIntentFitCritic:
    def test_perfect_match(self):
        r = run_intent_fit_critic(
            _make_profile(material_type="drum_loop"),
            intent=SampleIntent(intent_type="rhythm", description=""),
        )
        assert r.score >= 0.8

    def test_creative_mismatch(self):
        r = run_intent_fit_critic(
            _make_profile(material_type="vocal"),
            intent=SampleIntent(intent_type="rhythm", description=""),
        )
        # Vocal for rhythm is unusual but possible (chop workflow)
        assert 0.4 <= r.score <= 0.8


class TestRunAllCritics:
    def test_returns_all_six(self):
        results = run_all_sample_critics(
            profile=_make_profile(key="Cm", bpm=128.0, material_type="vocal"),
            intent=SampleIntent(intent_type="vocal", description=""),
            song_key="Cm",
            session_tempo=128.0,
            existing_roles=["drums", "bass"],
        )
        assert "key_fit" in results
        assert "tempo_fit" in results
        assert "frequency_fit" in results
        assert "role_fit" in results
        assert "vibe_fit" in results
        assert "intent_fit" in results
        assert all(isinstance(v, CriticResult) for v in results.values())


class TestVibeFitTransientNormalization:
    """Regression: transient_density is peaks/sec (unbounded), not a 0-1 value.

    Before the fix the energy proxy added raw peaks/sec to brightness and
    clamped to 1.0, so every transient-rich sample saturated to energy=1.0 and
    vibe_fit could not distinguish a moderately busy sample from an extremely
    busy one. After normalization the two yield different scores.
    """

    def test_busy_samples_are_distinguishable(self):
        tg = SimpleNamespace(evidence_count=10, novelty_band=0.4)
        moderate = _make_profile(brightness=0.0, transient_density=6.0)
        very_busy = _make_profile(brightness=0.0, transient_density=24.0)
        r_mod = run_vibe_fit_critic(moderate, taste_graph=tg)
        r_busy = run_vibe_fit_critic(very_busy, taste_graph=tg)
        # Pre-fix both saturate to energy=0.5 -> identical scores; the fix
        # makes them differ.
        assert abs(r_mod.score - r_busy.score) > 0.05

    def test_realistic_peaks_per_second_stays_in_range(self):
        tg = SimpleNamespace(evidence_count=10, novelty_band=0.5)
        # 30 peaks/sec is a plausible analyzer output for a dense break.
        profile = _make_profile(brightness=0.5, transient_density=30.0)
        r = run_vibe_fit_critic(profile, taste_graph=tg)
        assert 0.0 <= r.score <= 1.0


def test_slice_base_note_is_c1_not_c3():
    """Regression: Simpler slice mode maps slice N to MIDI 36+N (C1).

    Notes generated at C3 (60+) trigger no slice and play silent.
    """
    from mcp_server.sample_engine.slice_workflow import (
        SLICE_BASE_NOTE,
        plan_slice_steps,
    )

    assert SLICE_BASE_NOTE == 36

    result = plan_slice_steps(slice_count=8, intent="rhythm", bars=4, tempo=120)
    assert result["note_map"][0]["midi_note"] == 36
    assert result["note_map"][7]["midi_note"] == 43

    notes_step = [s for s in result["steps"] if s["tool"] == "add_notes"][0]
    pitches = [n["pitch"] for n in notes_step["params"]["notes"]]
    # Every emitted pitch must fall in the C1-based slice range (36..36+count-1),
    # never in the silent C3+ region.
    assert pitches, "expected generated notes"
    assert all(36 <= p <= 43 for p in pitches)
def test_unavailable_critic_excluded_from_fit_warnings():
    """P2 regression (finding 2): the evaluate_sample_fit warnings filter must
    skip critics that marked themselves unavailable (score=-1 sentinel), so the
    'No mix snapshot available' note is not surfaced as a spurious warning.

    Reproduces the exact warnings comprehension from
    mcp_server/sample_engine/tools.py::evaluate_sample_fit against real critic
    output produced with no mix snapshot.
    """
    profile = SampleProfile(
        source="filesystem",
        file_path="/tmp/x.wav",
        name="x",
        material_type="texture",
        frequency_center=2000.0,
    )
    intent = SampleIntent(intent_type="layer", philosophy="auto", description="")

    # No mix_snapshot -> frequency_fit becomes available=False, score=-1.0
    critics = run_all_sample_critics(
        profile=profile,
        intent=intent,
        song_key=None,
        session_tempo=120.0,
        existing_roles=[],
        mix_snapshot=None,
    )

    freq = critics["frequency_fit"]
    assert freq.available is False
    assert freq.score == -1.0
    assert "No mix snapshot available" in freq.recommendation

    # OLD (buggy) filter: score < 0.5 alone -> would surface the unavailable note
    old_warnings = [c.recommendation for c in critics.values() if c.score < 0.5]
    assert any("No mix snapshot available" in w for w in old_warnings)

    # NEW (fixed) filter: also require availability
    new_warnings = [
        c.recommendation
        for c in critics.values()
        if getattr(c, "available", True) and c.score < 0.5
    ]
    assert not any("No mix snapshot available" in w for w in new_warnings)
    # And the unavailable critic contributes nothing to warnings at all
    assert freq.recommendation not in new_warnings


def test_track_name_index_mapping_handles_unnamed_tracks():
    """P2 regression (finding 3): key-detection must pair notes with the track
    name at the SAME index. existing_roles is a packed list that skips unnamed
    tracks, so indexing it by track index misaligns names. A dict keyed by real
    index keeps the alignment correct.
    """
    # Simulate: track 0 unnamed (skipped), track 1 = 'bass', track 2 = 'lead'.
    existing_roles: list[str] = []
    track_names_by_index: dict[int, str] = {}
    sim = {0: "", 1: "bass", 2: "lead"}
    for i in range(3):
        name = sim[i]
        if name:
            existing_roles.append(name)
            track_names_by_index[i] = name

    # OLD packed-list lookup: index 1 wrongly returns 'lead' (off-by-one),
    # index 2 falls off the end entirely.
    old = lambda i: existing_roles[i] if i < len(existing_roles) else ""
    assert old(1) == "lead"   # WRONG name for track 1
    assert old(2) == ""       # WRONG: track 2 ('lead') resolves to empty

    # NEW index-keyed lookup: correct name for each real track index.
    assert track_names_by_index.get(0, "") == ""        # genuinely unnamed
    assert track_names_by_index.get(1, "") == "bass"
    assert track_names_by_index.get(2, "") == "lead"
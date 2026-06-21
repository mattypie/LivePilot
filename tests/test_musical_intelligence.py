"""Tests for musical intelligence detectors."""

import pytest

from mcp_server.musical_intelligence.detectors import (
    detect_repetition_fatigue,
    detect_role_conflicts,
    infer_section_purposes,
    score_emotional_arc,
    FatigueReport,
    RoleConflict,
    SectionPurpose,
    ArcScore,
)


# ═══ Repetition Fatigue ═══════════════════════════════════════════════

def test_no_fatigue_with_unique_clips():
    scenes = [
        {"name": "A", "clips": [{"name": "Clip 1", "state": "stopped"}]},
        {"name": "B", "clips": [{"name": "Clip 2", "state": "stopped"}]},
        {"name": "C", "clips": [{"name": "Clip 3", "state": "stopped"}]},
    ]
    report = detect_repetition_fatigue(scenes)
    assert report.fatigue_level < 0.3


def test_fatigue_with_overused_clips():
    clip = {"name": "Same Loop", "state": "stopped"}
    scenes = [
        {"name": "A", "clips": [clip]},
        {"name": "B", "clips": [clip]},
        {"name": "C", "clips": [clip]},
        {"name": "D", "clips": [clip]},
    ]
    report = detect_repetition_fatigue(scenes)
    assert report.fatigue_level > 0.2
    assert any(i["type"] == "clip_overuse" for i in report.issues)


def test_fatigue_with_motif_graph():
    scenes = [{"name": "A", "clips": []}]
    motif_graph = {
        "motifs": [
            {"motif_id": "motif_001", "fatigue_risk": 0.8},
            {"motif_id": "motif_002", "fatigue_risk": 0.3},
        ]
    }
    report = detect_repetition_fatigue(scenes, motif_graph)
    assert any(i["type"] == "motif_overuse" for i in report.issues)


def test_fatigue_empty_scenes():
    report = detect_repetition_fatigue([])
    assert report.fatigue_level == 0.0


def test_fatigue_recommendations():
    clip = {"name": "Loop", "state": "stopped"}
    scenes = [{"name": f"S{i}", "clips": [clip]} for i in range(5)]
    report = detect_repetition_fatigue(scenes)
    assert len(report.recommendations) > 0


# ═══ Role Conflicts ═══════════════════════════════════════════════════

def test_no_conflicts_clean_session():
    tracks = [
        {"index": 0, "name": "Drums"},
        {"index": 1, "name": "Sub Bass"},
        {"index": 2, "name": "Rhodes"},
        {"index": 3, "name": "Lead Synth"},
    ]
    conflicts = detect_role_conflicts(tracks)
    # Should have no conflicts (each role is unique)
    real_conflicts = [c for c in conflicts if len(c.tracks) > 1]
    assert len(real_conflicts) == 0


def test_bass_conflict():
    tracks = [
        {"index": 0, "name": "Sub Bass"},
        {"index": 1, "name": "808 Bass"},
        {"index": 2, "name": "Drums"},
    ]
    conflicts = detect_role_conflicts(tracks)
    bass_conflicts = [c for c in conflicts if c.role == "bass"]
    assert len(bass_conflicts) == 1
    assert len(bass_conflicts[0].tracks) == 2


def test_missing_essential_role():
    tracks = [
        {"index": 0, "name": "Synth Pad"},
        {"index": 1, "name": "Lead Melody"},
    ]
    conflicts = detect_role_conflicts(tracks)
    missing = [c for c in conflicts if len(c.tracks) == 0]
    assert len(missing) >= 1  # Missing bass and/or drums


def test_conflict_severity():
    tracks = [
        {"index": 0, "name": "Lead A"},
        {"index": 1, "name": "Lead B"},
        {"index": 2, "name": "Lead C"},
    ]
    conflicts = detect_role_conflicts(tracks)
    lead_conflict = [c for c in conflicts if c.role == "lead"]
    assert len(lead_conflict) == 1
    assert lead_conflict[0].severity > 0.5  # 3 leads = high severity


def test_bug_b1_drums_plus_perc_is_layering_not_conflict():
    """BUG-B1: DRUMS + PERC is the core aesthetic of hip-hop / Dilla /
    lo-fi — intentional layering, NOT a conflict. Severity should be
    demoted when one track is clearly a percussion extension."""
    tracks = [
        {"index": 0, "name": "DRUMS"},       # main kit
        {"index": 1, "name": "PERC"},        # percussion layer
    ]
    conflicts = detect_role_conflicts(tracks)
    drum_conflict = [c for c in conflicts if c.role == "drums"
                     and len(c.tracks) > 1]
    assert len(drum_conflict) == 1
    # Old behavior: severity=0.5. New behavior: demoted to ≤0.2 because
    # PERC is a percussion-layer keyword.
    assert drum_conflict[0].severity <= 0.2, (
        f"BUG-B1 regressed — DRUMS + PERC still flagged at severity "
        f"{drum_conflict[0].severity}"
    )
    # Recommendation must mention that the layering might be intentional
    assert (
        "intentional" in drum_conflict[0].recommendation.lower()
        or "hip-hop" in drum_conflict[0].recommendation.lower()
    )


def test_bug_b1_two_main_kits_still_flagged_severely():
    """Two tracks both named 'Drums' / 'DRUMS 2' — no percussion keyword —
    should still register as a real conflict (severity unchanged)."""
    tracks = [
        {"index": 0, "name": "Drums"},
        {"index": 1, "name": "Drums 2"},
    ]
    conflicts = detect_role_conflicts(tracks)
    drum_conflict = [c for c in conflicts if c.role == "drums"
                     and len(c.tracks) > 1]
    assert len(drum_conflict) == 1
    # Should stay at the original severity (0.3 + 0 * 0.2 = 0.3 or higher)
    assert drum_conflict[0].severity >= 0.3


# ═══ Section Purpose Inference ═══════════════════════════════════════

def _make_scene(name, active_clips, total=6):
    clips = [{"name": f"clip_{i}", "state": "stopped"} for i in range(active_clips)]
    clips += [{"state": "empty"} for _ in range(total - active_clips)]
    return {"name": name, "clips": clips}


def test_intro_detection():
    scenes = [
        _make_scene("Intro", 2),
        _make_scene("Full", 6),
    ]
    purposes = infer_section_purposes(scenes, total_tracks=6)
    assert purposes[0].purpose == "setup"


def test_payoff_detection():
    scenes = [
        _make_scene("Intro", 2),
        _make_scene("Build", 4),
        _make_scene("Drop", 6),
        _make_scene("Break", 2),
        _make_scene("Outro", 1),
    ]
    purposes = infer_section_purposes(scenes, total_tracks=6)
    # The full-density scene should be payoff
    high_energy = [p for p in purposes if p.energy >= 0.8]
    assert len(high_energy) >= 1


def test_contrast_detection():
    scenes = [
        _make_scene("Full", 6),
        _make_scene("Breakdown", 2),
        _make_scene("Return", 6),
    ]
    purposes = infer_section_purposes(scenes, total_tracks=6)
    contrasts = [p for p in purposes if p.purpose == "contrast"]
    assert len(contrasts) >= 1


def test_empty_scenes():
    purposes = infer_section_purposes([], total_tracks=6)
    assert len(purposes) == 0


# ═══ Emotional Arc Scoring ═══════════════════════════════════════════

def test_good_arc():
    sections = [
        SectionPurpose(name="Intro", purpose="setup", energy=0.3),
        SectionPurpose(name="Build", purpose="tension", energy=0.6),
        SectionPurpose(name="Drop", purpose="payoff", energy=0.9),
        SectionPurpose(name="Break", purpose="contrast", energy=0.4),
        SectionPurpose(name="Outro", purpose="release", energy=0.2),
    ]
    arc = score_emotional_arc(sections)
    assert arc.overall > 0.5
    assert arc.arc_clarity > 0.5
    assert arc.contrast > 0.5
    assert arc.resolution > 0.5


def test_flat_arc():
    sections = [
        SectionPurpose(name="A", purpose="development", energy=0.5),
        SectionPurpose(name="B", purpose="development", energy=0.5),
        SectionPurpose(name="C", purpose="development", energy=0.5),
    ]
    arc = score_emotional_arc(sections)
    assert arc.contrast < 0.3
    assert len(arc.issues) > 0
    assert any("contrast" in i.lower() for i in arc.issues)


def test_no_resolution():
    sections = [
        SectionPurpose(name="Build", purpose="tension", energy=0.6),
        SectionPurpose(name="Peak", purpose="payoff", energy=0.9),
        SectionPurpose(name="Still Peak", purpose="payoff", energy=0.9),
    ]
    arc = score_emotional_arc(sections)
    assert arc.resolution < 0.5
    assert any("resolution" in i.lower() for i in arc.issues)


def test_arc_empty():
    arc = score_emotional_arc([])
    assert arc.overall == 0.0
    assert len(arc.issues) > 0
def test_payoff_classified_as_payoff_not_tension():
    """A drop scene (peak density reached via a positive density jump) must
    be labeled 'payoff', not 'tension'. Before the fix the density-jump
    'tension' branch ran first and swallowed the drop."""
    scenes = [
        _make_scene("Intro", 2),   # density 0.333
        _make_scene("Build", 4),   # density 0.667, delta +0.333
        _make_scene("Drop", 6),    # density 1.0,   delta +0.333  <- the drop
    ]
    purposes = infer_section_purposes(scenes, total_tracks=6)
    drop = purposes[2]
    assert drop.density >= 0.8
    assert drop.purpose == "payoff", (
        f"Drop scene mislabeled as {drop.purpose!r} — high-density arrival "
        f"must classify as 'payoff', not 'tension'."
    )


def test_compare_phrase_renders_distinguishes_files(monkeypatch):
    """compare_phrase_renders must analyze each file so distinct audio
    produces distinct scores. Before the fix every render got an identical
    empty critique (overall == 0.333) and the ranking had zero signal."""
    from mcp_server.tools import _perception_engine as pe

    # Distinct loudness per file: flat dynamics vs a clear arc.
    fake_loudness = {
        "flat.wav": {"short_term_lufs": [-16, -16, -16], "lra_lu": 0.5},
        "arc.wav": {"short_term_lufs": [-20, -14, -18, -16, -15], "lra_lu": 5},
    }

    def fake_compute_loudness(path, detail="full"):
        name = path.split("/")[-1]
        return fake_loudness[name]

    def fake_compute_spectral(path):
        return None  # spectral not needed to differentiate the two

    monkeypatch.setattr(pe, "compute_loudness", fake_compute_loudness)
    monkeypatch.setattr(pe, "compute_spectral", fake_compute_spectral)

    from mcp_server.musical_intelligence import tools as mi_tools

    result = mi_tools.compare_phrase_renders(
        None, ["/tmp/flat.wav", "/tmp/arc.wav"], target="loop"
    )
    ranking = result["ranking"]
    assert len(ranking) == 2
    overalls = {r["render_id"]: r["overall"] for r in ranking}
    # The two files must NOT score identically — the bug made them equal.
    assert overalls["flat.wav"] != overalls["arc.wav"], (
        "compare_phrase_renders gave identical scores to different files — "
        "it is not analyzing the audio."
    )
    # And the file with the clearer arc should outrank the flat one.
    assert overalls["arc.wav"] > overalls["flat.wav"]
def test_fatigue_level_not_diluted_by_low_severity_issues():
    """A single serious issue must not be diluted when extra minor issues
    are added — fatigue aggregation is a saturating combine, not a mean.

    One motif at fatigue_risk 0.9 plus four at 0.61 each emits a
    `motif_overuse` issue (via the `fatigue_risk > 0.6` branch). The mean of
    [0.9, 0.61, 0.61, 0.61, 0.61] is ~0.668, so the old mean-based aggregation
    would report LESS fatigue than the single 0.9 issue warrants. With a
    saturating combine, more issues can only increase fatigue.
    """
    scenes = [{"name": "A", "clips": []}]
    motif_graph = {
        "motifs": [
            {"motif_id": "serious", "fatigue_risk": 0.9},
            {"motif_id": "minor_1", "fatigue_risk": 0.61},
            {"motif_id": "minor_2", "fatigue_risk": 0.61},
            {"motif_id": "minor_3", "fatigue_risk": 0.61},
            {"motif_id": "minor_4", "fatigue_risk": 0.61},
        ]
    }
    report = detect_repetition_fatigue(scenes, motif_graph)
    # Exactly the five motif_overuse issues, no others, for this input.
    assert len(report.issues) == 5
    assert all(i["type"] == "motif_overuse" for i in report.issues)
    # Saturating combine must keep fatigue at least as high as the worst issue
    # (0.9), not the diluted mean (~0.668).
    assert report.fatigue_level >= 0.9 - 1e-6
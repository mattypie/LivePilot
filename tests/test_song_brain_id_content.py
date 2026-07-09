"""Regression tests for content-aware brain_id and thread-safe snapshot store.

Problem: _compute_brain_id used to hash only {tempo, track_count,
scene_count}. A note-only edit (identical track/scene layout, different
melodic/harmonic/energy content) produced the SAME brain_id every time.
song_brain/tools.py keys its snapshot store (_brain_snapshots) by brain_id,
so an unchanged id meant every "after" build silently overwrote its own
"before" baseline and detect_identity_drift always read 0 drift.
"""

import threading
from types import SimpleNamespace

from mcp_server.song_brain import builder
from mcp_server.song_brain import tools as sb_tools
from mcp_server.song_brain.builder import build_song_brain, detect_identity_drift
from mcp_server.song_brain.models import SongBrain


def setup_function():
    sb_tools._brain_snapshots.clear()


# ── brain_id reflects content, not just structure ────────────────


def test_brain_id_changes_when_section_energy_shifts_same_structure():
    """Same tempo/track_count/scene_count, but section energies differ ->
    brain_id must differ (this was the core bug)."""
    session_info = {"tempo": 120, "track_count": 4}
    scenes = [{"name": "A", "clips": [1, 0]}, {"name": "B", "clips": [1, 1]}]

    comp_low_energy = {"sections": [
        {"name": "Intro", "id": "sec_00", "intent": "tension", "energy": 0.3},
        {"name": "Drop", "id": "sec_01", "intent": "payoff", "energy": 0.5},
    ]}
    comp_high_energy = {"sections": [
        {"name": "Intro", "id": "sec_00", "intent": "tension", "energy": 0.3},
        {"name": "Drop", "id": "sec_01", "intent": "payoff", "energy": 0.95},
    ]}

    before = build_song_brain(
        session_info=session_info, scenes=scenes, composition_analysis=comp_low_energy,
    )
    after = build_song_brain(
        session_info=session_info, scenes=scenes, composition_analysis=comp_high_energy,
    )

    # Structure (the old hash inputs) is identical...
    assert len(before.section_purposes) == len(after.section_purposes)
    # ...but the content-aware id must differ.
    assert before.brain_id != after.brain_id


def test_brain_id_changes_when_identity_core_differs_same_structure():
    session_info = {"tempo": 120, "track_count": 3}
    tracks_a = [{"name": "Vocal Hook", "index": 0}]
    tracks_b = [{"name": "Pad Lush", "index": 0}]

    brain_a = build_song_brain(session_info=session_info, tracks=tracks_a)
    brain_b = build_song_brain(session_info=session_info, tracks=tracks_b)

    assert brain_a.identity_core != brain_b.identity_core
    assert brain_a.brain_id != brain_b.brain_id


def test_brain_id_stable_across_identical_builds():
    """Same inputs -> same id, every time (still a pure deterministic fn)."""
    kwargs = dict(
        session_info={"tempo": 128, "track_count": 5},
        scenes=[{"name": "Intro", "clips": [1, 0]}, {"name": "Drop", "clips": [1, 1]}],
        tracks=[{"name": "Kick", "index": 0}],
        motif_data={"motifs": [{"name": "hook", "salience": 0.7, "description": "Lead melody"}]},
    )
    brain1 = build_song_brain(**kwargs)
    brain2 = build_song_brain(**kwargs)
    assert brain1.brain_id == brain2.brain_id


def test_brain_id_still_differs_on_structure_change():
    """Sanity: the original structural signal (track_count) still matters."""
    brain_a = build_song_brain(session_info={"tempo": 120, "track_count": 4})
    brain_b = build_song_brain(session_info={"tempo": 120, "track_count": 8})
    assert brain_a.brain_id != brain_b.brain_id


# ── Content-aware brain_id prevents snapshot self-collision ──────


def test_brain_id_prevents_snapshot_collision_on_content_only_change():
    """End-to-end regression: before the fix, two builds with identical
    track/scene counts but different section energy got the SAME brain_id
    and silently clobbered each other's entry in _brain_snapshots, so
    detect_identity_drift compared a brain against itself (drift=0) even
    though the song's content had genuinely changed."""
    ctx = SimpleNamespace(lifespan_context={})

    session_info = {"tempo": 120, "track_count": 4}
    scenes = [{"name": "A"}, {"name": "B"}]
    comp1 = {"sections": [{"name": "Drop", "id": "sec_00", "intent": "payoff", "energy": 0.9}]}
    comp2 = {"sections": [{"name": "Drop", "id": "sec_00", "intent": "payoff", "energy": 0.3}]}

    before = builder.build_song_brain(
        session_info=session_info, scenes=scenes, composition_analysis=comp1,
    )
    sb_tools._set_brain(ctx, before)

    after = builder.build_song_brain(
        session_info=session_info, scenes=scenes, composition_analysis=comp2,
    )
    sb_tools._set_brain(ctx, after)

    assert before.brain_id != after.brain_id
    # Both snapshots coexist — the "before" wasn't silently overwritten.
    assert sb_tools._get_snapshot(before.brain_id) is before
    assert sb_tools._get_snapshot(after.brain_id) is after

    drift = detect_identity_drift(before, after)
    assert drift.energy_arc_shift > 0, "drift must be visible now that ids don't collide"


# ── Snapshot store concurrency ─────────────────────────────────────


def test_brain_snapshot_store_and_evict_is_thread_safe():
    """Concurrent _set_brain calls must not corrupt _brain_snapshots (e.g. a
    'dictionary changed size during iteration' RuntimeError from the
    store-then-evict read-modify-write) and must respect _MAX_SNAPSHOTS."""
    errors: list[Exception] = []

    def worker(i: int) -> None:
        try:
            ctx = SimpleNamespace(lifespan_context={})
            sb_tools._set_brain(ctx, SongBrain(brain_id=f"brain_{i}"))
        except Exception as exc:  # pragma: no cover - failure path only
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(200)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent _set_brain raised: {errors}"
    assert len(sb_tools._brain_snapshots) <= sb_tools._MAX_SNAPSHOTS

"""Handler-level tests for remote_script.LivePilot.notes.

Covers: add_notes loop_end auto-extension (BUG-2026-04-22#1c), get_notes
field shape (modern-API dict per note), remove_notes / remove_notes_by_id,
modify_notes' two-pass fail-all-or-apply-all validation, duplicate_notes,
transpose_notes' MIDI-range clamping (0-127, skip not clip), and
quantize_clip.

notes.py does `import Live` INSIDE add_notes/duplicate_notes (not at
module scope) to construct `Live.Clip.MidiNoteSpecification(...)`. We
still stub `Live` before loading the module (matching the project's
established pattern) and provide a real `Live.Clip.MidiNoteSpecification`
class so those inline imports resolve to a usable fake.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = ROOT / "remote_script" / "LivePilot"


class _FakeMidiNoteSpecification:
    """Stand-in for Live.Clip.MidiNoteSpecification — a plain attr bag."""

    def __init__(self, pitch, start_time, duration, velocity=100.0,
                 mute=False, probability=1.0, velocity_deviation=0.0,
                 release_velocity=64.0):
        self.pitch = pitch
        self.start_time = start_time
        self.duration = duration
        self.velocity = velocity
        self.mute = mute
        self.probability = probability
        self.velocity_deviation = velocity_deviation
        self.release_velocity = release_velocity
        self.note_id = None  # assigned by the fake clip on add


def _install_fake_live_module():
    live_mod = types.ModuleType("Live")
    clip_mod = types.ModuleType("Live.Clip")
    clip_mod.MidiNoteSpecification = _FakeMidiNoteSpecification
    live_mod.Clip = clip_mod
    sys.modules["Live"] = live_mod
    sys.modules["Live.Clip"] = clip_mod
    return live_mod


def _load_remote_notes():
    for name in [
        "remote_script.LivePilot.notes",
        "remote_script.LivePilot._clip_helpers",
        "remote_script.LivePilot.router",
        "remote_script.LivePilot.utils",
        "remote_script.LivePilot",
        "remote_script",
    ]:
        sys.modules.pop(name, None)

    _install_fake_live_module()

    remote_pkg = types.ModuleType("remote_script")
    remote_pkg.__path__ = [str(ROOT / "remote_script")]
    sys.modules["remote_script"] = remote_pkg

    live_pkg = types.ModuleType("remote_script.LivePilot")
    live_pkg.__path__ = [str(REMOTE_ROOT)]
    sys.modules["remote_script.LivePilot"] = live_pkg

    def _load(name: str, path: Path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    _load("remote_script.LivePilot.utils", REMOTE_ROOT / "utils.py")
    router = _load("remote_script.LivePilot.router", REMOTE_ROOT / "router.py")
    _load("remote_script.LivePilot._clip_helpers", REMOTE_ROOT / "_clip_helpers.py")
    notes = _load("remote_script.LivePilot.notes", REMOTE_ROOT / "notes.py")
    return router, notes


class _FakeMidiClip:
    """A minimal fake exercising the modern note API surface notes.py uses."""

    def __init__(self, length=4.0, loop_end=4.0, end_marker=4.0):
        self.length = length
        self.loop_end = loop_end
        self.end_marker = end_marker
        self._next_id = 1
        self._notes = []  # list of _FakeMidiNoteSpecification-like objects

    def add_new_notes(self, note_specs):
        for spec in note_specs:
            spec.note_id = self._next_id
            self._next_id += 1
            self._notes.append(spec)

    def get_notes_extended(self, from_pitch, pitch_span, from_time, time_span):
        out = []
        for n in self._notes:
            if from_pitch <= n.pitch < from_pitch + pitch_span:
                if from_time <= n.start_time < from_time + time_span:
                    out.append(n)
        return out

    def remove_notes_extended(self, from_pitch, pitch_span, from_time, time_span):
        keep = []
        for n in self._notes:
            in_range = (from_pitch <= n.pitch < from_pitch + pitch_span and
                        from_time <= n.start_time < from_time + time_span)
            if not in_range:
                keep.append(n)
        self._notes = keep

    def remove_notes_by_id(self, ids):
        id_set = set(ids)
        self._notes = [n for n in self._notes if n.note_id not in id_set]

    def apply_note_modifications(self, notes):
        # In the real API this writes back in-place mutations already
        # applied to the shared NoteVector objects — our fake notes are
        # already mutated in place by the handler, so this is a no-op
        # marker call we can assert was invoked via a counter if needed.
        pass


def _song_with_clip(clip):
    slot = types.SimpleNamespace(clip=clip)
    track = types.SimpleNamespace(clip_slots=[slot], arrangement_clips=[])

    class _Song:
        tracks = [track]
        return_tracks = []
        master_track = None
        undo_steps = 0

        def begin_undo_step(self):
            self.undo_steps += 1

        def end_undo_step(self):
            self.undo_steps -= 1

    return _Song()


def test_add_notes_extends_loop_end_when_note_exceeds_clip_length():
    """BUG-2026-04-22#1c: a note landing past loop_end must auto-extend it,
    and the response reports loop_end_extended_to."""
    _router, notes_mod = _load_remote_notes()
    clip = _FakeMidiClip(length=4.0, loop_end=4.0, end_marker=4.0)
    song = _song_with_clip(clip)

    result = notes_mod.add_notes(song, {
        "track_index": 0, "clip_index": 0,
        "notes": [{"pitch": 60, "start_time": 6.0, "duration": 1.0, "velocity": 100}],
    })
    assert result["notes_added"] == 1
    assert result["loop_end_extended_to"] == pytest.approx(7.0)
    assert clip.loop_end == pytest.approx(7.0)
    assert clip.end_marker == pytest.approx(7.0)


def test_add_notes_within_loop_end_does_not_extend():
    _router, notes_mod = _load_remote_notes()
    clip = _FakeMidiClip(length=4.0, loop_end=4.0, end_marker=4.0)
    song = _song_with_clip(clip)

    result = notes_mod.add_notes(song, {
        "track_index": 0, "clip_index": 0,
        "notes": [{"pitch": 60, "start_time": 0.0, "duration": 1.0}],
    })
    assert "loop_end_extended_to" not in result
    assert clip.loop_end == pytest.approx(4.0)


def test_add_notes_rejects_empty_list():
    _router, notes_mod = _load_remote_notes()
    clip = _FakeMidiClip()
    song = _song_with_clip(clip)

    with pytest.raises(ValueError, match="cannot be empty"):
        notes_mod.add_notes(song, {"track_index": 0, "clip_index": 0, "notes": []})


def test_get_notes_returns_modern_api_field_shape():
    _router, notes_mod = _load_remote_notes()
    clip = _FakeMidiClip()
    song = _song_with_clip(clip)
    notes_mod.add_notes(song, {
        "track_index": 0, "clip_index": 0,
        "notes": [{"pitch": 64, "start_time": 0.5, "duration": 0.25,
                   "velocity": 90, "probability": 0.8}],
    })

    result = notes_mod.get_notes(song, {"track_index": 0, "clip_index": 0})
    assert len(result["notes"]) == 1
    note = result["notes"][0]
    assert set(note.keys()) == {
        "note_id", "pitch", "start_time", "duration", "velocity",
        "mute", "probability", "velocity_deviation", "release_velocity",
    }
    assert note["pitch"] == 64
    assert note["probability"] == pytest.approx(0.8)


def test_remove_notes_by_range():
    _router, notes_mod = _load_remote_notes()
    clip = _FakeMidiClip()
    song = _song_with_clip(clip)
    notes_mod.add_notes(song, {
        "track_index": 0, "clip_index": 0,
        "notes": [{"pitch": 60, "start_time": 0.0, "duration": 1.0},
                  {"pitch": 72, "start_time": 2.0, "duration": 1.0}],
    })

    notes_mod.remove_notes(song, {
        "track_index": 0, "clip_index": 0,
        "from_pitch": 60, "pitch_span": 1, "from_time": 0.0, "time_span": 1.0,
    })
    remaining = notes_mod.get_notes(song, {"track_index": 0, "clip_index": 0})
    assert len(remaining["notes"]) == 1
    assert remaining["notes"][0]["pitch"] == 72


def test_remove_notes_by_id_removes_only_requested():
    _router, notes_mod = _load_remote_notes()
    clip = _FakeMidiClip()
    song = _song_with_clip(clip)
    notes_mod.add_notes(song, {
        "track_index": 0, "clip_index": 0,
        "notes": [{"pitch": 60, "start_time": 0.0, "duration": 1.0},
                  {"pitch": 61, "start_time": 1.0, "duration": 1.0}],
    })
    first_id = clip._notes[0].note_id

    result = notes_mod.remove_notes_by_id(song, {
        "track_index": 0, "clip_index": 0, "note_ids": [first_id],
    })
    assert result["removed_count"] == 1
    remaining = notes_mod.get_notes(song, {"track_index": 0, "clip_index": 0})
    assert len(remaining["notes"]) == 1
    assert remaining["notes"][0]["pitch"] == 61


def test_modify_notes_fails_all_when_any_note_id_missing():
    """Two-pass validation: an unknown note_id must abort BEFORE mutating
    any note — no partial application."""
    _router, notes_mod = _load_remote_notes()
    clip = _FakeMidiClip()
    song = _song_with_clip(clip)
    notes_mod.add_notes(song, {
        "track_index": 0, "clip_index": 0,
        "notes": [{"pitch": 60, "start_time": 0.0, "duration": 1.0}],
    })
    real_id = clip._notes[0].note_id

    with pytest.raises(ValueError, match="No modifications applied"):
        notes_mod.modify_notes(song, {
            "track_index": 0, "clip_index": 0,
            "modifications": [
                {"note_id": real_id, "pitch": 72},
                {"note_id": 999999, "pitch": 40},
            ],
        })
    # The valid entry must NOT have been applied either (fail-all).
    unchanged = notes_mod.get_notes(song, {"track_index": 0, "clip_index": 0})
    assert unchanged["notes"][0]["pitch"] == 60


def test_modify_notes_applies_all_when_valid():
    _router, notes_mod = _load_remote_notes()
    clip = _FakeMidiClip()
    song = _song_with_clip(clip)
    notes_mod.add_notes(song, {
        "track_index": 0, "clip_index": 0,
        "notes": [{"pitch": 60, "start_time": 0.0, "duration": 1.0}],
    })
    real_id = clip._notes[0].note_id

    result = notes_mod.modify_notes(song, {
        "track_index": 0, "clip_index": 0,
        "modifications": [{"note_id": real_id, "pitch": 72, "velocity": 111}],
    })
    assert result["modified_count"] == 1
    updated = notes_mod.get_notes(song, {"track_index": 0, "clip_index": 0})
    assert updated["notes"][0]["pitch"] == 72
    assert updated["notes"][0]["velocity"] == 111


def test_duplicate_notes_applies_time_offset():
    _router, notes_mod = _load_remote_notes()
    clip = _FakeMidiClip()
    song = _song_with_clip(clip)
    notes_mod.add_notes(song, {
        "track_index": 0, "clip_index": 0,
        "notes": [{"pitch": 60, "start_time": 0.0, "duration": 1.0}],
    })
    real_id = clip._notes[0].note_id

    result = notes_mod.duplicate_notes(song, {
        "track_index": 0, "clip_index": 0,
        "note_ids": [real_id], "time_offset": 4.0,
    })
    assert result["duplicated_count"] == 1
    all_notes = notes_mod.get_notes(song, {
        "track_index": 0, "clip_index": 0, "time_span": 100.0,
    })
    start_times = sorted(n["start_time"] for n in all_notes["notes"])
    assert start_times == [0.0, 4.0]


def test_duplicate_notes_no_matching_ids_raises():
    _router, notes_mod = _load_remote_notes()
    clip = _FakeMidiClip()
    song = _song_with_clip(clip)

    with pytest.raises(ValueError, match="No matching notes"):
        notes_mod.duplicate_notes(song, {
            "track_index": 0, "clip_index": 0, "note_ids": [12345],
        })


def test_transpose_notes_clamps_out_of_midi_range():
    """Notes that would exceed 0-127 after transposition are skipped, not
    clamped or wrapped, and the response reports the skip count."""
    _router, notes_mod = _load_remote_notes()
    clip = _FakeMidiClip()
    song = _song_with_clip(clip)
    notes_mod.add_notes(song, {
        "track_index": 0, "clip_index": 0,
        "notes": [{"pitch": 120, "start_time": 0.0, "duration": 1.0},
                  {"pitch": 60, "start_time": 1.0, "duration": 1.0}],
    })

    result = notes_mod.transpose_notes(song, {
        "track_index": 0, "clip_index": 0, "semitones": 12,
        "time_span": 100.0,
    })
    assert result["transposed_count"] == 1
    assert result["skipped_out_of_range"] == 1
    assert "exceed MIDI range" in result["warning"]

    final = notes_mod.get_notes(song, {
        "track_index": 0, "clip_index": 0, "time_span": 100.0,
    })
    pitches = sorted(n["pitch"] for n in final["notes"])
    assert pitches == [72, 120]  # 60+12 transposed; 120+12=132 skipped, stays 120


def test_transpose_notes_negative_below_zero_skipped():
    _router, notes_mod = _load_remote_notes()
    clip = _FakeMidiClip()
    song = _song_with_clip(clip)
    notes_mod.add_notes(song, {
        "track_index": 0, "clip_index": 0,
        "notes": [{"pitch": 5, "start_time": 0.0, "duration": 1.0}],
    })

    result = notes_mod.transpose_notes(song, {
        "track_index": 0, "clip_index": 0, "semitones": -10,
        "time_span": 100.0,
    })
    assert result["transposed_count"] == 0
    assert result["skipped_out_of_range"] == 1


def test_quantize_clip_calls_clip_quantize_with_grid_and_amount():
    _router, notes_mod = _load_remote_notes()

    class _QuantizeClip(_FakeMidiClip):
        def __init__(self):
            super().__init__()
            self.quantize_calls = []

        def quantize(self, grid, amount):
            self.quantize_calls.append((grid, amount))

    clip = _QuantizeClip()
    song = _song_with_clip(clip)

    result = notes_mod.quantize_clip(song, {
        "track_index": 0, "clip_index": 0, "grid": 5, "amount": 0.75,
    })
    assert result["quantized"] is True
    assert clip.quantize_calls == [(5, 0.75)]

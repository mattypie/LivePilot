"""Tests for the semantic move compiler — verifies moves compile to concrete tool calls."""

import pytest

from mcp_server.semantic_moves.models import SemanticMove
from mcp_server.semantic_moves.compiler import compile, CompiledPlan, CompiledStep
from mcp_server.semantic_moves.registry import get_move
from mcp_server.semantic_moves import resolvers


# ── Mock kernel ──────────────────────────────────────────────────────────────

MOCK_KERNEL = {
    "session_info": {
        "tempo": 82,
        "track_count": 6,
        "tracks": [
            {"index": 0, "name": "Drums", "mute": False, "solo": False},
            {"index": 1, "name": "Sub Bass", "mute": False, "solo": False},
            {"index": 2, "name": "Rhodes", "mute": False, "solo": False},
            {"index": 3, "name": "Texture Pad", "mute": False, "solo": False},
            {"index": 4, "name": "Glitch Lead", "mute": False, "solo": False},
            {"index": 5, "name": "Lo-fi Perc", "mute": False, "solo": False},
        ],
    },
    "mode": "improve",
    "capability_state": {},
}


# ── Resolver tests ───────────────────────────────────────────────────────────

def test_infer_role_drums():
    assert resolvers.infer_role("Drums") == "drums"
    assert resolvers.infer_role("808 Kit") == "drums"  # "kit" matches drums first
    assert resolvers.infer_role("Main Beat") == "drums"


def test_infer_role_bass():
    assert resolvers.infer_role("Sub Bass") == "bass"
    assert resolvers.infer_role("Deep Sub") == "bass"


def test_infer_role_chords():
    assert resolvers.infer_role("Rhodes") == "chords"
    assert resolvers.infer_role("Piano Chords") == "chords"


def test_infer_role_pad():
    assert resolvers.infer_role("Texture Pad") == "pad"
    assert resolvers.infer_role("Ambient Drone") == "pad"


def test_infer_role_lead():
    assert resolvers.infer_role("Glitch Lead") == "lead"
    assert resolvers.infer_role("Synth Melody") == "lead"


def test_infer_role_unknown():
    assert resolvers.infer_role("Track 7") == "unknown"


def test_find_tracks_by_role():
    bass = resolvers.find_tracks_by_role(MOCK_KERNEL, ["bass"])
    assert len(bass) == 1
    assert bass[0]["name"] == "Sub Bass"
    assert bass[0]["index"] == 1


def test_find_tracks_by_role_multiple():
    melodic = resolvers.find_tracks_by_role(MOCK_KERNEL, ["chords", "lead"])
    assert len(melodic) == 2
    names = {t["name"] for t in melodic}
    assert "Rhodes" in names
    assert "Glitch Lead" in names


def test_find_track_by_name():
    t = resolvers.find_track_by_name(MOCK_KERNEL, "Rhodes")
    assert t is not None
    assert t["index"] == 2


def test_find_track_by_name_not_found():
    assert resolvers.find_track_by_name(MOCK_KERNEL, "Violin") is None


def test_volume_math():
    assert resolvers.clamp_volume(1.5) == 1.0
    assert resolvers.clamp_volume(-0.5) == 0.0
    assert resolvers.adjust_volume(0.7, 5) == pytest.approx(0.75)
    assert resolvers.adjust_volume(0.95, 10) == 1.0  # Clamped


# ── Compiler tests ───────────────────────────────────────────────────────────

def test_compile_make_punchier():
    move = get_move("make_punchier")
    assert move is not None
    plan = compile(move, MOCK_KERNEL)
    assert isinstance(plan, CompiledPlan)
    assert plan.executable
    assert plan.move_id == "make_punchier"
    # Should have steps for drums + pads + verification
    assert plan.step_count >= 3
    # All steps must have a tool name
    for step in plan.steps:
        assert step.tool, f"Step missing tool: {step.description}"


def test_compile_tighten_low_end():
    move = get_move("tighten_low_end")
    plan = compile(move, MOCK_KERNEL)
    assert plan.executable
    # Should target Sub Bass track
    bass_steps = [s for s in plan.steps if "Sub Bass" in s.description]
    assert len(bass_steps) >= 1


def test_compile_widen_stereo():
    move = get_move("widen_stereo")
    plan = compile(move, MOCK_KERNEL)
    assert plan.executable
    pan_steps = [s for s in plan.steps if s.tool == "set_track_pan"]
    assert len(pan_steps) >= 2  # At least chords + lead


def test_compile_darken_mix():
    move = get_move("darken_without_losing_width")
    plan = compile(move, MOCK_KERNEL)
    assert plan.executable


def test_compile_reduce_repetition():
    move = get_move("reduce_repetition_fatigue")
    plan = compile(move, MOCK_KERNEL)
    assert plan.executable
    perlin_steps = [s for s in plan.steps if "perlin" in s.description.lower()]
    assert len(perlin_steps) >= 1


def test_compile_unknown_move_returns_non_executable():
    fake_move = SemanticMove(move_id="nonexistent", family="unknown", intent="???")
    plan = compile(fake_move, MOCK_KERNEL)
    assert not plan.executable
    assert len(plan.warnings) > 0


def test_compiled_plan_to_dict():
    move = get_move("make_punchier")
    plan = compile(move, MOCK_KERNEL)
    d = plan.to_dict()
    assert "steps" in d
    assert "summary" in d
    assert "executable" in d
    assert d["executable"] is True
    for step in d["steps"]:
        assert "tool" in step
        assert "params" in step
        assert "description" in step


def test_compiled_plan_requires_approval_in_improve_mode():
    move = get_move("make_punchier")
    plan = compile(move, MOCK_KERNEL)
    assert plan.requires_approval is True


def test_compiled_plan_auto_executes_in_explore_mode():
    move = get_move("make_punchier")
    kernel = {**MOCK_KERNEL, "mode": "explore"}
    plan = compile(move, kernel)
    assert plan.requires_approval is False


def test_empty_session_degrades_gracefully():
    """Compiler should handle sessions with no tracks."""
    empty_kernel = {
        "session_info": {"tempo": 120, "track_count": 0, "tracks": []},
        "mode": "improve",
    }
    move = get_move("make_punchier")
    plan = compile(move, empty_kernel)
    # Should still return a plan (maybe with warnings)
    assert isinstance(plan, CompiledPlan)
    assert len(plan.warnings) > 0 or plan.step_count <= 2  # Just reads + verify


# ── LIVE#4 — make_kick_bass_lock must emit EQ + sidechain, not just volume ──

def test_compile_make_kick_bass_lock_has_eq_step():
    """LIVE#4: compiled plan must include an EQ insert or EQ-param step, not
    solely a volume cut. A volume-only plan is too shallow to count as
    'frequency separation'."""
    move = get_move("make_kick_bass_lock")
    assert move is not None
    plan = compile(move, MOCK_KERNEL)
    assert plan.executable

    tools_used = [s.tool for s in plan.steps]
    # Must contain a device insertion or device-parameter step
    has_eq_action = (
        "insert_device" in tools_used
        or any(
            s.tool == "set_device_parameter" and "eq" in s.description.lower()
            or s.tool == "set_device_parameter" and "hp" in s.description.lower()
            or s.tool == "set_device_parameter" and "high" in s.description.lower()
            or s.tool == "set_device_parameter" and "filter" in s.description.lower()
            or s.tool == "set_device_parameter" and "freq" in s.description.lower()
            for s in plan.steps
        )
    )
    assert has_eq_action, (
        "make_kick_bass_lock compiled plan must include EQ insert or EQ-param step. "
        f"Tools used: {tools_used}"
    )


def test_compile_make_kick_bass_lock_has_sidechain_step():
    """LIVE#4: compiled plan must do sidechain-compression work — either route
    an existing compressor's sidechain, or insert a Compressor scaffold when
    none is resolvable. Not just static frequency carving."""
    move = get_move("make_kick_bass_lock")
    plan = compile(move, MOCK_KERNEL)

    # MOCK_KERNEL's bass has no devices → compressor inserted as a scaffold.
    has_sidechain_action = any(
        s.tool == "compressor_set_sidechain"
        or (s.tool == "insert_device"
            and "compressor" in str(s.params.get("device_name", "")).lower())
        for s in plan.steps
    )
    assert has_sidechain_action, (
        "make_kick_bass_lock must include sidechain work (compressor_set_sidechain "
        f"or a Compressor insert). Tools used: {[s.tool for s in plan.steps]}"
    )


def test_compile_make_kick_bass_lock_sidechain_targets_kick():
    """LIVE#4: when a compressor is resolvable on the bass, the sidechain step
    must use the REAL compressor_set_sidechain contract (track_index,
    device_index, source_type, source_channel) and key from the kick track."""
    kernel = {
        "session_info": {
            "tempo": 120,
            "track_count": 2,
            "tracks": [
                {"index": 0, "name": "Drums", "mute": False, "solo": False},
                {"index": 1, "name": "Sub Bass", "mute": False, "solo": False,
                 "devices": [
                     {"index": 0, "name": "Compressor", "class_name": "Compressor2"},
                 ]},
            ],
        },
        "mode": "improve",
        "capability_state": {},
    }
    move = get_move("make_kick_bass_lock")
    plan = compile(move, kernel)

    sc_steps = [s for s in plan.steps if s.tool == "compressor_set_sidechain"]
    assert sc_steps, "Expected a compressor_set_sidechain step when a compressor is resolvable"
    sc = sc_steps[0]
    # Real tool signature — routes the detector input, no invented params.
    assert sc.params.get("device_index") == 0
    assert "source_type" in sc.params
    # source must reference the kick/drum track by display name.
    assert sc.params["source_type"] == "Drums"
    # The wrong/invented param from the original buggy fix must be gone.
    assert "sidechain_track_index" not in sc.params


def test_compile_make_kick_bass_lock_volume_is_not_only_step():
    """LIVE#4: a volume-only plan (set_track_volume as sole mutating step) is
    the buggy behaviour — the plan must do more than just lower volume."""
    move = get_move("make_kick_bass_lock")
    plan = compile(move, MOCK_KERNEL)

    mutating_steps = [
        s for s in plan.steps
        if s.tool not in ("get_track_meters", "get_master_spectrum", "get_device_parameters")
    ]
    volume_only = all(s.tool == "set_track_volume" for s in mutating_steps)
    assert not volume_only, (
        "make_kick_bass_lock must not compile to a volume-only plan. "
        f"Mutating tools: {[s.tool for s in mutating_steps]}"
    )


def test_compile_make_kick_bass_lock_no_kick_track_degrades():
    """LIVE#4: when no kick track exists, sidechain step is omitted gracefully."""
    no_kick_kernel = {
        "session_info": {
            "tempo": 120,
            "track_count": 2,
            "tracks": [
                {"index": 0, "name": "Sub Bass", "mute": False, "solo": False},
                {"index": 1, "name": "Pad", "mute": False, "solo": False},
            ],
        },
        "mode": "improve",
        "capability_state": {},
    }
    move = get_move("make_kick_bass_lock")
    plan = compile(move, no_kick_kernel)
    assert isinstance(plan, CompiledPlan)
    # Should warn about missing kick
    assert any("kick" in w.lower() or "drum" in w.lower() for w in plan.warnings)
    # Must not crash — steps can still include EQ work
    for step in plan.steps:
        assert step.tool, f"Step missing tool: {step.description}"


# ── LIVE#5 — widen_stereo fallback when no lead/harmonic roles found ──

# Kernel with tracks named in a musical-role style that doesn't match keywords
MOCK_KERNEL_UNLABELED = {
    "session_info": {
        "tempo": 122,
        "track_count": 6,
        "tracks": [
            {"index": 0, "name": "Kick", "mute": False, "solo": False},
            {"index": 1, "name": "FMBass", "mute": False, "solo": False},
            {"index": 2, "name": "Q-Call", "mute": False, "solo": False},   # no lead/chord keyword
            {"index": 3, "name": "A-Answer", "mute": False, "solo": False}, # no lead/chord keyword
            {"index": 4, "name": "Atmos", "mute": False, "solo": False},
            {"index": 5, "name": "PercFX", "mute": False, "solo": False},
        ],
    },
    "mode": "improve",
    "capability_state": {},
}


def test_compile_widen_stereo_fallback_produces_pan_steps():
    """LIVE#5: widen_stereo must produce real pan steps even when no
    lead/harmonic tracks are role-classified. The bug was a no-op plan
    with 0 pan steps and summary 'No panning changes'."""
    move = get_move("widen_stereo")
    assert move is not None
    plan = compile(move, MOCK_KERNEL_UNLABELED)
    assert plan.executable

    pan_steps = [s for s in plan.steps if s.tool == "set_track_pan"]
    assert len(pan_steps) >= 1, (
        "widen_stereo must emit at least one set_track_pan step even when no "
        "lead/harmonic role is classified. Got 0 pan steps. "
        f"Summary: {plan.summary!r}. Warnings: {plan.warnings}"
    )


def test_compile_widen_stereo_fallback_excludes_bass_and_drums():
    """LIVE#5: fallback widening must not pan bass or drum tracks — those
    should stay mono/centered."""
    move = get_move("widen_stereo")
    plan = compile(move, MOCK_KERNEL_UNLABELED)

    pan_steps = [s for s in plan.steps if s.tool == "set_track_pan"]
    # Drum track is index 0 (Kick), bass is index 1 (FMBass)
    excluded_indices = {0, 1}
    for step in pan_steps:
        ti = step.params.get("track_index")
        assert ti not in excluded_indices, (
            f"set_track_pan step targets track {ti} which is a drum/bass track "
            "and should not be panned by widen_stereo. "
            f"Step: {step.description!r}"
        )


def test_compile_widen_stereo_fallback_warns():
    """LIVE#5: fallback path must emit a warning explaining why it fell back,
    so the agent knows the role-classification gap."""
    move = get_move("widen_stereo")
    plan = compile(move, MOCK_KERNEL_UNLABELED)

    assert any(
        "fallback" in w.lower() or "fall back" in w.lower() or "falling back" in w.lower()
        for w in plan.warnings
    ), (
        "widen_stereo fallback must include a warning explaining the fallback. "
        f"Got warnings: {plan.warnings}"
    )


def test_compile_widen_stereo_primary_path_unchanged():
    """LIVE#5: primary path (with proper role-named tracks) must still work."""
    move = get_move("widen_stereo")
    plan = compile(move, MOCK_KERNEL)
    assert plan.executable
    pan_steps = [s for s in plan.steps if s.tool == "set_track_pan"]
    assert len(pan_steps) >= 2  # Chords + lead


def test_compile_widen_stereo_no_tracks_at_all():
    """LIVE#5: widen_stereo with zero non-drum/non-bass tracks must warn but not crash."""
    move = get_move("widen_stereo")
    plan = compile(move, {
        "session_info": {
            "tempo": 120,
            "track_count": 2,
            "tracks": [
                {"index": 0, "name": "Kick", "mute": False, "solo": False},
                {"index": 1, "name": "Bass", "mute": False, "solo": False},
            ],
        },
        "mode": "improve",
        "capability_state": {},
    })
    assert isinstance(plan, CompiledPlan)
    # No crash, but should have a warning
    assert len(plan.warnings) >= 1


# ── LIVE#4+5 verify-after guard — zero meters when stopped ──────────────────

def _make_ctx_with_session(is_playing: bool = False):
    """Build a fake Context that returns session_info with is_playing set."""
    from types import SimpleNamespace

    class _FakeExecResult:
        def __init__(self, tool, ok, result, error=None):
            self.tool = tool
            self.backend = "remote_command"
            self.ok = ok
            self.result = result
            self.error = error

    class _FakeAbleton:
        def __init__(self, playing):
            self._playing = playing

        def send_command(self, cmd, params=None):
            if cmd == "get_session_info":
                return {
                    "tempo": 120,
                    "track_count": 2,
                    "tracks": [
                        {"index": 0, "name": "Kick", "mute": False},
                        {"index": 1, "name": "Sub Bass", "mute": False},
                    ],
                    "is_playing": self._playing,
                }
            return {}

    return SimpleNamespace(lifespan_context={"ableton": _FakeAbleton(is_playing)})


def test_verify_after_stopped_transport_does_not_inflate_success():
    """LIVE#4+5: when get_track_meters returns is_playing=False, those steps
    must NOT be counted in success_count.

    We test the guard logic directly by simulating the executed_steps structure
    that apply_semantic_move builds, then applying the same detection heuristic.
    """
    # Simulate the executed_steps list that the move would produce when
    # playback is stopped: the EQ/sidechain steps succeed (ok=True),
    # but the final get_track_meters also "succeeds" yet returns stopped data.
    meter_result_stopped = {
        "is_playing": False,
        "tracks": [
            {"index": 0, "peak": 0.0, "rms": 0.0},
            {"index": 1, "peak": 0.0, "rms": 0.0},
        ],
    }

    executed_steps = [
        {"tool": "insert_device", "ok": True, "result": {"ok": True},
         "description": "Insert EQ Eight", "backend": "remote_command"},
        {"tool": "set_device_parameter", "ok": True, "result": {"ok": True},
         "description": "HP filter", "backend": "remote_command"},
        {"tool": "compressor_set_sidechain", "ok": True, "result": {"ok": True},
         "description": "Sidechain", "backend": "remote_command"},
        {"tool": "get_track_meters", "ok": True, "result": meter_result_stopped,
         "description": "Verify after", "backend": "remote_command"},
    ]

    # Apply the same guard logic as in tools.py
    _METER_VERIFY_TOOLS = {"get_track_meters", "get_master_meters"}
    meter_verify_skipped_count = 0
    for es in executed_steps:
        if es["tool"] not in _METER_VERIFY_TOOLS:
            continue
        if not es["ok"] or es["result"] is None:
            continue
        result_data = es["result"]
        is_playing_flag = result_data.get("is_playing") if isinstance(result_data, dict) else None
        if is_playing_flag is False:
            es["verification_skipped"] = True
            es["verification_note"] = "Playback stopped"
            meter_verify_skipped_count += 1

    success_count = sum(
        1 for s in executed_steps
        if s["ok"] and not s.get("verification_skipped", False)
    )

    # The meter step must have been detected as skipped
    assert meter_verify_skipped_count == 1, (
        f"Expected 1 skipped meter verify step, got {meter_verify_skipped_count}"
    )
    # success_count must NOT include the skipped meter step
    assert success_count == 3, (
        f"Expected success_count=3 (3 real steps), got {success_count}. "
        "The zero-meter verify step must not be counted as a success."
    )
    # The skipped step must be annotated
    skipped = [s for s in executed_steps if s.get("verification_skipped")]
    assert len(skipped) == 1
    assert skipped[0]["tool"] == "get_track_meters"


def test_verify_after_playing_transport_counts_meter_step():
    """Complementary: when is_playing=True the meter step IS counted normally."""
    meter_result_playing = {
        "is_playing": True,
        "tracks": [
            {"index": 0, "peak": 0.72, "rms": 0.58},
            {"index": 1, "peak": 0.65, "rms": 0.51},
        ],
    }

    executed_steps = [
        {"tool": "insert_device", "ok": True, "result": {"ok": True},
         "description": "Insert EQ Eight", "backend": "remote_command"},
        {"tool": "get_track_meters", "ok": True, "result": meter_result_playing,
         "description": "Verify after", "backend": "remote_command"},
    ]

    _METER_VERIFY_TOOLS = {"get_track_meters", "get_master_meters"}
    meter_verify_skipped_count = 0
    for es in executed_steps:
        if es["tool"] not in _METER_VERIFY_TOOLS:
            continue
        if not es["ok"] or es["result"] is None:
            continue
        result_data = es["result"]
        is_playing_flag = result_data.get("is_playing") if isinstance(result_data, dict) else None
        if is_playing_flag is False:
            es["verification_skipped"] = True
            meter_verify_skipped_count += 1

    success_count = sum(
        1 for s in executed_steps
        if s["ok"] and not s.get("verification_skipped", False)
    )

    # Nothing skipped — playback was running
    assert meter_verify_skipped_count == 0
    # Both steps count
    assert success_count == 2


def test_verify_after_real_path_flags_zero_meters_without_is_playing():
    """LIVE45-1: drive the REAL apply_semantic_move explore path. The meter
    verify step runs as a remote_command whose result has NO is_playing key
    (bare remote shape) with all-zero meters → the fallback must flag it
    verification_skipped so it does not inflate success. (The other verify-after
    tests re-implement the guard inline; this exercises the router end-to-end.)"""
    import asyncio
    from types import SimpleNamespace
    from mcp_server.semantic_moves.tools import apply_semantic_move
    from mcp_server.semantic_moves.models import SemanticMove
    from mcp_server.semantic_moves.registry import register, _REGISTRY
    from mcp_server.semantic_moves import compiler as move_compiler
    from mcp_server.semantic_moves.compiler import CompiledPlan, CompiledStep

    move_id = "_verify_zero_probe"
    register(SemanticMove(move_id=move_id, family="mix", intent="verify probe"))

    def _probe(move, kernel):
        return CompiledPlan(
            move_id=move.move_id, intent=move.intent,
            steps=[CompiledStep(tool="get_track_meters",
                                params={"include_stereo": True},
                                description="verify", backend="remote_command")],
            summary="probe",
        )
    move_compiler.register_compiler(move_id, _probe)

    class _FakeAbleton:
        def send_command(self, command, params=None):
            if command == "get_track_meters":
                # Bare remote shape: NO is_playing key, all-zero meters.
                return {"tracks": [{"index": 0, "level": 0.0, "left": 0.0, "right": 0.0}]}
            if command == "get_session_info":
                return {"tempo": 120, "tracks": [], "scenes": []}
            return {}

    try:
        ctx = SimpleNamespace(lifespan_context={"ableton": _FakeAbleton()})
        result = asyncio.run(apply_semantic_move(ctx, move_id=move_id, mode="explore"))
        assert result.get("verification_skipped_count", 0) >= 1, (
            f"zero-meter verify step not flagged: {result.get('execution_results')}")
    finally:
        _REGISTRY.pop(move_id, None)
        move_compiler._COMPILERS.pop(move_id, None)

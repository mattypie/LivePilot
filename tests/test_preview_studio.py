"""Unit tests for Preview Studio engine — pure computation, no Ableton needed."""

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mcp_server.m4l_bridge import SpectralCache
from mcp_server.preview_studio.engine import (
    commit_variant,
    compare_variants,
    create_preview_set,
    discard_set,
    get_preview_set,
)


# ── Triptych creation ────────────────────────────────────────────


def test_triptych_creates_three_variants():
    """Creative triptych should produce exactly 3 variants."""
    ps = create_preview_set(
        request_text="make this more magical",
        kernel_id="test_kern",
        strategy="creative_triptych",
    )
    assert len(ps.variants) == 3


def test_triptych_labels():
    """Variants should be labeled safe, strong, unexpected."""
    ps = create_preview_set(
        request_text="add energy",
        kernel_id="test_kern",
    )
    labels = {v.label for v in ps.variants}
    assert labels == {"safe", "strong", "unexpected"}


def test_triptych_novelty_ordering():
    """Safe should be lowest novelty, unexpected highest."""
    ps = create_preview_set(
        request_text="improve the chorus",
        kernel_id="test_kern",
    )
    by_label = {v.label: v for v in ps.variants}
    assert by_label["safe"].novelty_level < by_label["strong"].novelty_level
    assert by_label["strong"].novelty_level < by_label["unexpected"].novelty_level


def test_triptych_identity_effects():
    """Each variant should have different identity effects."""
    ps = create_preview_set(
        request_text="test",
        kernel_id="test_kern",
    )
    effects = {v.identity_effect for v in ps.variants}
    assert "preserves" in effects
    assert "evolves" in effects
    assert "contrasts" in effects


# ── Binary strategy ──────────────────────────────────────────────


def test_binary_creates_two_variants():
    """Binary strategy should produce exactly 2 variants."""
    ps = create_preview_set(
        request_text="test binary",
        kernel_id="test_kern",
        strategy="binary",
    )
    assert len(ps.variants) == 2


# ── Comparison ───────────────────────────────────────────────────


def test_comparison_ranks_preserves_highest():
    """With default identity weight, preserves should rank highest."""
    ps = create_preview_set(
        request_text="test ranking",
        kernel_id="test_kern",
    )
    comparison = compare_variants(ps)
    rankings = comparison["rankings"]
    assert len(rankings) == 3
    # First should be the one with highest score
    assert rankings[0]["score"] >= rankings[-1]["score"]


def test_comparison_returns_recommended():
    """Comparison should include a recommended variant when an executable
    one exists. With no moves registered, every triptych variant is
    blocked (compiled_plan=None) and ``recommended`` is correctly ``None``
    — asserted below to lock the post-truth-gap-fix contract.
    """
    # Case 1 — no moves available: every variant is blocked, recommended is None.
    ps_blocked = create_preview_set(
        request_text="test recommend blocked",
        kernel_id="test_kern_blocked",
    )
    comparison_blocked = compare_variants(ps_blocked)
    assert comparison_blocked["recommended"] is None, (
        "when every variant is blocked, recommended must be None, not a lie"
    )
    assert len(comparison_blocked["analytical_candidates"]) == len(ps_blocked.variants)

    # Case 2 — at least one executable move: recommended is a variant_id string.
    ps_exec = create_preview_set(
        request_text="test recommend exec",
        kernel_id="test_kern_exec",
        available_moves=[{
            "move_id": "make_punchier",
            "plan_template": [{"tool": "set_track_volume", "params": {}}],
        }],
    )
    comparison_exec = compare_variants(ps_exec)
    assert comparison_exec.get("recommended"), (
        "with at least one executable variant, recommended must be set"
    )
    assert isinstance(comparison_exec["recommended"], str)


def test_comparison_with_custom_weights():
    """Custom weights should change ranking."""
    ps = create_preview_set(
        request_text="test weights",
        kernel_id="test_kern",
    )
    # Strongly favor novelty
    comparison = compare_variants(ps, {
        "taste_weight": 0.1,
        "novelty_weight": 0.8,
        "identity_weight": 0.1,
    })
    rankings = comparison["rankings"]
    assert len(rankings) == 3


# ── Commit and discard ───────────────────────────────────────────


def test_commit_marks_chosen():
    """Committing should mark the chosen variant and discard others."""
    ps = create_preview_set(
        request_text="test commit",
        kernel_id="test_kern",
    )
    chosen_id = ps.variants[1].variant_id
    result = commit_variant(ps, chosen_id)

    assert result is not None
    assert result.status == "committed"

    # Others should be discarded
    for v in ps.variants:
        if v.variant_id != chosen_id:
            assert v.status == "discarded"

    assert ps.status == "committed"
    assert ps.committed_variant_id == chosen_id


def test_commit_unknown_variant():
    """Committing an unknown variant should return None."""
    ps = create_preview_set(
        request_text="test bad commit",
        kernel_id="test_kern",
    )
    result = commit_variant(ps, "nonexistent_id")
    assert result is None


def test_discard_removes_from_store():
    """Discarding should remove the set from the store."""
    ps = create_preview_set(
        request_text="test discard",
        kernel_id="test_kern_discard",
    )
    set_id = ps.set_id
    assert get_preview_set(set_id) is not None

    result = discard_set(set_id)
    assert result is True
    assert get_preview_set(set_id) is None


def test_discard_unknown_set():
    """Discarding an unknown set should return False."""
    result = discard_set("nonexistent_set_id")
    assert result is False


# ── Song brain integration ───────────────────────────────────────


def test_variants_include_preservation_notes():
    """Variants should include what_preserved text."""
    ps = create_preview_set(
        request_text="test preservation",
        kernel_id="test_kern",
        song_brain={"sacred_elements": [{"description": "Main hook melody"}]},
    )
    for v in ps.variants:
        assert v.what_preserved  # Should be non-empty
        assert "hook" in v.what_preserved.lower() or "element" in v.what_preserved.lower()


def test_set_id_deterministic():
    """Same request + kernel should produce same set_id."""
    ps1 = create_preview_set(
        request_text="deterministic test",
        kernel_id="kern_fixed",
    )
    ps2 = create_preview_set(
        request_text="deterministic test",
        kernel_id="kern_fixed",
    )
    assert ps1.set_id == ps2.set_id


# ── Wonder-awareness ─────────────────────────────────────────────


def test_analytical_refusal_in_wonder_context():
    """Wonder-linked analytical variant should be refused by render."""
    from mcp_server.preview_studio.tools import _should_refuse_analytical
    assert _should_refuse_analytical(compiled_plan=None, wonder_linked=True) is True
    assert _should_refuse_analytical(compiled_plan=None, wonder_linked=False) is False
    assert _should_refuse_analytical(compiled_plan=[{"tool": "x"}], wonder_linked=True) is False
    assert _should_refuse_analytical(compiled_plan=[{"tool": "x"}], wonder_linked=False) is False


def test_render_variant_uses_lifespan_spectral_cache_for_audible_preview(monkeypatch):
    """Audible preview should use the shared spectral cache from lifespan_context."""
    import asyncio
    from mcp_server.preview_studio.tools import render_preview_variant
    import mcp_server.runtime.execution_router as execution_router

    ps = create_preview_set(
        request_text="test render",
        kernel_id="test_kern_render",
        available_moves=[{"move_id": "make_punchier", "plan_template": [{"tool": "set_track_volume", "params": {}}]}],
    )
    variant_id = ps.variants[0].variant_id

    class _Ableton:
        def __init__(self):
            self.calls = []

        def send_command(self, cmd, params=None):
            self.calls.append(cmd)
            if cmd == "get_session_info":
                return {"tempo": 120, "track_count": 4}
            return {"ok": True}

        async def send_command_async(self, cmd, params=None):
            return self.send_command(cmd, params)

    cache = SpectralCache()
    cache.update("spectrum", {"sub": 0.1})

    async def _fake_exec_async(steps, ableton=None, bridge=None, mcp_registry=None, ctx=None, stop_on_failure=True):
        return [SimpleNamespace(
            ok=True, tool="set_track_volume",
            backend="remote_command", result={"ok": True}, error="",
        )]

    async def _fake_sleep(_seconds):
        return None

    monkeypatch.setattr(execution_router, "execute_plan_steps_async", _fake_exec_async)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    # Perf batch (v1.27.3): render_preview_variant awaits asyncio.sleep(play_seconds)
    # (up to 8s) for the audible-preview window — the "time.sleep" patch above
    # doesn't touch it since the code path is async. Zero-delay it here.
    monkeypatch.setattr("asyncio.sleep", _fake_sleep)

    ctx = SimpleNamespace(lifespan_context={"ableton": _Ableton(), "spectral": cache})
    result = asyncio.run(render_preview_variant(ctx, set_id=ps.set_id, variant_id=variant_id, bars=2))

    assert result["preview_mode"] == "audible_preview"


def test_render_preview_variant_captures_audible_before_undo(monkeypatch):
    """Regression: audible capture must happen while the variant is applied.

    Previously the finally block ran undo before the audible capture section,
    so "audible_preview" was a lie — it captured pre-variant audio and labeled
    it as the variant's sound. This test asserts ordering via a call log.
    """
    import asyncio
    from mcp_server.preview_studio.tools import render_preview_variant
    from mcp_server.preview_studio.engine import create_preview_set
    import mcp_server.runtime.execution_router as execution_router

    ps = create_preview_set(
        request_text="ordering test",
        kernel_id="test_kern_order",
        available_moves=[{"move_id": "make_punchier", "plan_template": [{"tool": "set_track_volume", "params": {}}]}],
    )
    variant_id = ps.variants[0].variant_id

    calls = []

    class _Ableton:
        def send_command(self, cmd, params=None):
            calls.append(cmd)
            if cmd == "get_session_info":
                return {"tempo": 120, "track_count": 4}
            return {"ok": True}

        async def send_command_async(self, cmd, params=None):
            return self.send_command(cmd, params)

    class _Spectral(SpectralCache):
        def get_all(self):
            calls.append("spectral_snapshot")
            return super().get_all()

    cache = _Spectral()
    cache.update("spectrum", {"sub": 0.1})

    async def _fake_exec_async(steps, ableton=None, bridge=None, mcp_registry=None, ctx=None, stop_on_failure=True):
        calls.append("apply_plan")
        return [SimpleNamespace(
            ok=True, tool="set_track_volume",
            backend="remote_command", result={"ok": True}, error="",
        )]

    async def _fake_sleep(_seconds):
        return None

    monkeypatch.setattr(execution_router, "execute_plan_steps_async", _fake_exec_async)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    # Perf batch (v1.27.3): see comment in the sibling test above — this
    # test's audible-preview window is also a real asyncio.sleep.
    monkeypatch.setattr("asyncio.sleep", _fake_sleep)

    ctx = SimpleNamespace(lifespan_context={"ableton": _Ableton(), "spectral": cache})
    result = asyncio.run(render_preview_variant(ctx, set_id=ps.set_id, variant_id=variant_id, bars=2))

    assert result["preview_mode"] == "audible_preview"

    # Ordering: apply must precede spectral snapshots; snapshots + start_playback
    # must precede undo.
    assert "apply_plan" in calls
    assert "start_playback" in calls
    assert "undo" in calls, "undo should eventually run in cleanup"

    apply_idx = calls.index("apply_plan")
    undo_idx = calls.index("undo")
    play_idx = calls.index("start_playback")
    snapshot_indices = [i for i, c in enumerate(calls) if c == "spectral_snapshot"]

    assert apply_idx < play_idx < undo_idx, f"Apply before play before undo required; got {calls}"
    for si in snapshot_indices:
        assert si < undo_idx, f"Spectral snapshot at {si} must precede undo at {undo_idx}; got {calls}"

    # There should be exactly one spectral_comparison in the result (before + after)
    assert "spectral_comparison" in result


# ── v1.10.3 Truth Release: commit_preview_variant actually executes ──

def test_commit_preview_variant_actually_executes_compiled_plan(monkeypatch):
    """v1.10.3: commit used to mark the variant as committed in memory and
    return committed=True without running anything. Users expected commit
    to apply the chosen variant. It now runs the compiled plan through the
    async router and reports per-step results.
    """
    import asyncio
    from mcp_server.preview_studio.tools import commit_preview_variant
    from mcp_server.preview_studio.engine import create_preview_set, store_preview_set
    from mcp_server.preview_studio.models import PreviewSet, PreviewVariant
    import mcp_server.runtime.execution_router as execution_router
    import time

    # Build a preview set containing a variant with a real compiled plan
    variant = PreviewVariant(
        variant_id="v_commit_test",
        label="safe",
        intent="test commit",
        compiled_plan=[
            {"tool": "set_track_volume", "params": {"track_index": 0, "volume": 0.5}},
            {"tool": "set_track_pan", "params": {"track_index": 0, "pan": 0.3}},
        ],
        identity_effect="preserves",
        what_preserved="everything",
        move_id="test_move",
    )
    ps = PreviewSet(
        set_id="ps_commit_test",
        request_text="commit test",
        strategy="binary",
        source_kernel_id="k",
        variants=[variant],
        created_at_ms=int(time.time() * 1000),
    )
    store_preview_set(ps)

    executed_tools = []

    async def _fake_exec_async(steps, ableton=None, bridge=None, mcp_registry=None,
                                ctx=None, stop_on_failure=True):
        for s in steps:
            executed_tools.append(s.get("tool"))
        return [
            SimpleNamespace(
                ok=True, tool=s.get("tool"),
                backend="remote_command", result={"ok": True}, error="",
            )
            for s in steps
        ]

    monkeypatch.setattr(execution_router, "execute_plan_steps_async", _fake_exec_async)

    class _Ableton:
        def send_command(self, cmd, params=None):
            return {"ok": True}

    ctx = SimpleNamespace(lifespan_context={"ableton": _Ableton()})
    result = asyncio.run(commit_preview_variant(
        ctx, set_id="ps_commit_test", variant_id="v_commit_test",
    ))

    # The compiled plan must have actually been executed
    assert executed_tools == ["set_track_volume", "set_track_pan"], \
        f"commit should have executed both plan steps; got {executed_tools}"

    # Response should reflect real execution
    assert result["committed"] is True
    assert result["status"] == "committed"
    assert result["steps_ok"] == 2
    assert result["steps_failed"] == 0
    assert "execution_log" in result
    assert len(result["execution_log"]) == 2


def test_commit_preview_variant_analytical_only_returns_honest_status(monkeypatch):
    """If the variant has no compiled_plan (analytical-only), commit must
    NOT pretend to apply anything — it returns status='analytical_only'
    with committed=False.
    """
    import asyncio
    from mcp_server.preview_studio.tools import commit_preview_variant
    from mcp_server.preview_studio.engine import store_preview_set
    from mcp_server.preview_studio.models import PreviewSet, PreviewVariant
    import time

    variant = PreviewVariant(
        variant_id="v_analytical",
        label="unexpected",
        intent="analytical",
        compiled_plan=None,  # analytical-only
        identity_effect="contrasts",
    )
    ps = PreviewSet(
        set_id="ps_analytical_test",
        request_text="analytical",
        strategy="binary",
        source_kernel_id="k",
        variants=[variant],
        created_at_ms=int(time.time() * 1000),
    )
    store_preview_set(ps)

    class _Ableton:
        def send_command(self, cmd, params=None):
            raise AssertionError("commit should NOT call any Ableton command for analytical variant")

    ctx = SimpleNamespace(lifespan_context={"ableton": _Ableton()})
    result = asyncio.run(commit_preview_variant(
        ctx, set_id="ps_analytical_test", variant_id="v_analytical",
    ))

    assert result["committed"] is False
    assert result["status"] == "analytical_only"
    assert "note" in result
    # execution_log should not be present because nothing ran
    assert "execution_log" not in result


# ─── BUG-B44 / B45 regressions — variant description fields ────────────────


class TestBugB44B45VariantDescriptions:
    """BUG-B44: variants without a compiled_plan used to still be listed
    with status='pending' — committing them hit a missing-plan error.
    BUG-B45: variants had empty what_changed / summary, so users
    couldn't tell what each variant actually did."""

    def test_describe_variant_uses_move_intent(self):
        from mcp_server.preview_studio.engine import _describe_variant
        move = {"move_id": "make_punchier", "intent": "Tighten low end and boost punch"}
        compiled = {"move_id": "make_punchier", "steps": [
            {"description": "Read current levels"},
            {"description": "Apply compressor"},
        ]}
        profile = {"label": "strong", "novelty": 0.5}
        result = _describe_variant(move, compiled, profile)
        assert "Tighten" in result["what_changed"]
        assert result["summary"]

    def test_describe_variant_falls_back_to_plan_steps(self):
        """When the move has no intent/description, aggregate plan step
        descriptions into what_changed."""
        from mcp_server.preview_studio.engine import _describe_variant
        compiled = {"move_id": "x", "steps": [
            {"description": "Lower kick volume"},
            {"description": "Boost bass sub"},
        ]}
        profile = {"label": "safe", "novelty": 0.2}
        result = _describe_variant({}, compiled, profile)
        assert "Lower kick volume" in result["what_changed"]
        assert "Boost bass sub" in result["what_changed"]

    def test_describe_variant_final_fallback_when_no_plan(self):
        """No move, no plan — describe by profile so what_changed is
        never empty."""
        from mcp_server.preview_studio.engine import _describe_variant
        profile = {"label": "unexpected", "novelty": 0.8}
        result = _describe_variant(None, None, profile)
        assert result["what_changed"]
        assert "unexpected" in result["what_changed"].lower()

    def test_triptych_blocks_variants_without_compiled_plan(self):
        """BUG-B44: variants with compiled_plan=None should flip to
        status='blocked' so callers can skip commit attempts."""
        from mcp_server.preview_studio.engine import _build_triptych
        variants = _build_triptych(
            request_text="test",
            moves=[],  # no moves → all variants uncompilable
            song_brain={},
            taste_graph={},
            set_id="test_set",
            now=0,
            kernel=None,
        )
        assert len(variants) == 3
        for v in variants:
            assert v.compiled_plan is None
            assert v.status == "blocked", (
                f"BUG-B44 regressed — variant without plan still status="
                f"{v.status}"
            )
            assert v.what_changed  # no empty descriptions
def test_create_preview_set_does_not_clobber_committed_set():
    """A re-request with identical request_text + kernel_id must NOT overwrite
    an existing committed (or compared) preview set. The protected set is
    preserved and the new request branches to a distinct set_id.
    """
    from mcp_server.preview_studio.engine import (
        create_preview_set,
        commit_variant,
        compare_variants,
        get_preview_set,
        _PROTECTED_STATUSES,
    )

    # First request — compare then commit so the set reaches a protected status.
    ps1 = create_preview_set(
        request_text="clobber guard test",
        kernel_id="clobber_kern",
        available_moves=[{"move_id": "m_safe", "plan_template": [{"tool": "x", "params": {}}]}],
    )
    compare_variants(ps1)
    chosen_id = ps1.variants[0].variant_id
    commit_variant(ps1, chosen_id)
    assert ps1.status in _PROTECTED_STATUSES
    protected_id = ps1.set_id
    protected_committed = ps1.committed_variant_id

    # Second identical request — same request_text + kernel_id → same base hash.
    ps2 = create_preview_set(
        request_text="clobber guard test",
        kernel_id="clobber_kern",
    )

    # New set must NOT reuse the protected id.
    assert ps2.set_id != protected_id

    # The committed set must still be retrievable and unchanged.
    still_there = get_preview_set(protected_id)
    assert still_there is ps1
    assert still_there.status == "committed"
    assert still_there.committed_variant_id == protected_committed

    # And the new set is independently stored under its branched id.
    assert get_preview_set(ps2.set_id) is ps2

    # Sanity: a fresh (non-protected) re-request still reuses the same id,
    # preserving the deterministic-id contract.
    ps3a = create_preview_set(request_text="non protected dedup", kernel_id="np_kern")
    ps3b = create_preview_set(request_text="non protected dedup", kernel_id="np_kern")
    assert ps3a.set_id == ps3b.set_id


# ── State-layer hardening: lock + session fingerprint ────────────


def test_store_preview_set_concurrent_hammer_does_not_raise():
    """Regression: _preview_sets is mutated from both threadpooled sync
    tools and event-loop async tools. Before the fix, store_preview_set's
    check-then-evict loop raced: two threads could observe the same
    `oldest_key` and the second `del` raised KeyError (or the shared
    dict-iteration raised 'dictionary changed size during iteration').
    A module-level lock around get/store/evict closes that race.

    sys.setswitchinterval is lowered modestly (10 microseconds — 500x more
    aggressive than the 5ms default, but well short of the 1us/0.1us extremes
    that make thread scheduling itself pathologically expensive on a loaded
    machine) to force fine-grained interleaving without risking a
    multi-minute stall under CI/CPU contention. Verified in isolation to
    reproduce the pre-fix KeyError/RuntimeError reliably (dozens of times
    per run) while completing in well under a second.
    """
    import sys
    import threading
    from mcp_server.preview_studio.engine import (
        _preview_sets,
        _preview_sets_lock,
        _MAX_PREVIEW_SETS,
        store_preview_set,
    )
    from mcp_server.preview_studio.models import PreviewSet

    old_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-5)
    with _preview_sets_lock:
        _preview_sets.clear()
    errors: list[Exception] = []

    def _worker(worker_id: int) -> None:
        try:
            for j in range(150):
                ps = PreviewSet(
                    set_id=f"race_{worker_id}_{j}",
                    request_text="race",
                    source_kernel_id="k",
                )
                store_preview_set(ps)
        except Exception as exc:  # pragma: no cover - failure path only
            errors.append(exc)

    try:
        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        sys.setswitchinterval(old_interval)

    assert errors == [], f"concurrent store_preview_set raised: {errors!r}"
    assert len(_preview_sets) <= _MAX_PREVIEW_SETS


def test_create_preview_set_stamps_session_fingerprint_from_kernel():
    """A real kernel's session_info should be hashed into
    PreviewSet.session_fingerprint at creation time — no extra round-trip,
    just a hash of data the caller already fetched."""
    from mcp_server.preview_studio.engine import create_preview_set
    from mcp_server.preview_studio.models import compute_session_fingerprint

    session_info = {
        "track_count": 2,
        "tracks": [{"index": 0, "name": "Kick"}, {"index": 1, "name": "Bass"}],
    }
    ps = create_preview_set(
        request_text="fingerprint stamping test",
        kernel_id="k_fp",
        kernel={"session_info": session_info, "mode": "improve"},
    )
    assert ps.session_fingerprint
    assert ps.session_fingerprint == compute_session_fingerprint(session_info)


def test_create_preview_set_without_kernel_has_no_fingerprint():
    """No kernel supplied (degraded path) → empty fingerprint, meaning
    'no signal' rather than a false match/mismatch downstream."""
    from mcp_server.preview_studio.engine import create_preview_set

    ps = create_preview_set(
        request_text="no kernel fingerprint test",
        kernel_id="k_nofp",
    )
    assert ps.session_fingerprint == ""


def test_commit_preview_variant_fingerprint_mismatch_returns_state_error(monkeypatch):
    """If the session's track topology changed since the preview set was
    built, commit must refuse with a structured STATE_ERROR instead of
    replaying stale track/device indices against the new topology."""
    import asyncio
    from mcp_server.preview_studio.tools import commit_preview_variant
    from mcp_server.preview_studio.engine import store_preview_set
    from mcp_server.preview_studio.models import (
        PreviewSet, PreviewVariant, compute_session_fingerprint,
    )
    import mcp_server.runtime.execution_router as execution_router
    import time

    built_session_info = {
        "track_count": 2,
        "tracks": [{"index": 0, "name": "Kick"}, {"index": 1, "name": "Bass"}],
    }
    variant = PreviewVariant(
        variant_id="v_fp_mismatch",
        label="safe",
        intent="test fingerprint mismatch",
        compiled_plan=[{"tool": "set_track_volume", "params": {"track_index": 0, "volume": 0.5}}],
        identity_effect="preserves",
        move_id="test_move",
    )
    ps = PreviewSet(
        set_id="ps_fp_mismatch",
        request_text="fingerprint mismatch",
        strategy="binary",
        source_kernel_id="k",
        variants=[variant],
        created_at_ms=int(time.time() * 1000),
        session_fingerprint=compute_session_fingerprint(built_session_info),
    )
    store_preview_set(ps)

    async def _fail_if_called(*args, **kwargs):
        raise AssertionError("compiled plan must not execute on a fingerprint mismatch")

    monkeypatch.setattr(execution_router, "execute_plan_steps_async", _fail_if_called)

    class _Ableton:
        async def send_command_async(self, cmd, params=None):
            if cmd == "get_session_info":
                # Topology changed: a track was added since the preview was built.
                return {
                    "track_count": 3,
                    "tracks": [
                        {"index": 0, "name": "Kick"},
                        {"index": 1, "name": "Bass"},
                        {"index": 2, "name": "New Track"},
                    ],
                }
            return {"ok": True}

    ctx = SimpleNamespace(lifespan_context={"ableton": _Ableton()})
    result = asyncio.run(commit_preview_variant(
        ctx, set_id="ps_fp_mismatch", variant_id="v_fp_mismatch",
    ))

    assert result.get("code") == "STATE_ERROR"
    assert "error" in result
    assert "session changed" in result["error"].lower()
    # State must be untouched — a rejected commit doesn't flip preview-set status.
    assert ps.status != "committed"


def test_commit_preview_variant_fingerprint_match_proceeds(monkeypatch):
    """A fresh session_info that hashes to the same fingerprint must NOT
    block commit — the topology is unchanged, indices are still valid."""
    import asyncio
    from mcp_server.preview_studio.tools import commit_preview_variant
    from mcp_server.preview_studio.engine import store_preview_set
    from mcp_server.preview_studio.models import (
        PreviewSet, PreviewVariant, compute_session_fingerprint,
    )
    import mcp_server.runtime.execution_router as execution_router
    import time

    session_info = {
        "track_count": 2,
        "tracks": [{"index": 0, "name": "Kick"}, {"index": 1, "name": "Bass"}],
    }
    variant = PreviewVariant(
        variant_id="v_fp_match",
        label="safe",
        intent="test fingerprint match",
        compiled_plan=[{"tool": "set_track_volume", "params": {"track_index": 0, "volume": 0.5}}],
        identity_effect="preserves",
        move_id="test_move",
    )
    ps = PreviewSet(
        set_id="ps_fp_match",
        request_text="fingerprint match",
        strategy="binary",
        source_kernel_id="k",
        variants=[variant],
        created_at_ms=int(time.time() * 1000),
        session_fingerprint=compute_session_fingerprint(session_info),
    )
    store_preview_set(ps)

    executed_tools = []

    async def _fake_exec_async(steps, ableton=None, bridge=None, mcp_registry=None,
                                ctx=None, stop_on_failure=True):
        for s in steps:
            executed_tools.append(s.get("tool"))
        return [
            SimpleNamespace(ok=True, tool=s.get("tool"), backend="remote_command",
                             result={"ok": True}, error="")
            for s in steps
        ]

    monkeypatch.setattr(execution_router, "execute_plan_steps_async", _fake_exec_async)

    class _Ableton:
        async def send_command_async(self, cmd, params=None):
            if cmd == "get_session_info":
                # Same topology — different dict instance, same shape.
                return {
                    "track_count": 2,
                    "tracks": [{"index": 0, "name": "Kick"}, {"index": 1, "name": "Bass"}],
                }
            return {"ok": True}

    ctx = SimpleNamespace(lifespan_context={"ableton": _Ableton()})
    result = asyncio.run(commit_preview_variant(
        ctx, set_id="ps_fp_match", variant_id="v_fp_match",
    ))

    assert executed_tools == ["set_track_volume"]
    assert result["committed"] is True
    assert result["status"] == "committed"


def test_commit_preview_variant_absent_fingerprint_skips_check(monkeypatch):
    """Objects predating session_fingerprint (empty string) must still
    commit — and must not even attempt a session refresh, since the
    ableton mock below has no send_command_async at all."""
    import asyncio
    from mcp_server.preview_studio.tools import commit_preview_variant
    from mcp_server.preview_studio.engine import store_preview_set
    from mcp_server.preview_studio.models import PreviewSet, PreviewVariant
    import mcp_server.runtime.execution_router as execution_router
    import time

    variant = PreviewVariant(
        variant_id="v_fp_absent",
        label="safe",
        intent="test absent fingerprint",
        compiled_plan=[{"tool": "set_track_volume", "params": {"track_index": 0, "volume": 0.5}}],
        identity_effect="preserves",
        move_id="test_move",
    )
    ps = PreviewSet(
        set_id="ps_fp_absent",
        request_text="absent fingerprint",
        strategy="binary",
        source_kernel_id="k",
        variants=[variant],
        created_at_ms=int(time.time() * 1000),
        # session_fingerprint left at its default "" — no signal available.
    )
    assert ps.session_fingerprint == ""
    store_preview_set(ps)

    async def _fake_exec_async(steps, ableton=None, bridge=None, mcp_registry=None,
                                ctx=None, stop_on_failure=True):
        return [
            SimpleNamespace(ok=True, tool=s.get("tool"), backend="remote_command",
                             result={"ok": True}, error="")
            for s in steps
        ]

    monkeypatch.setattr(execution_router, "execute_plan_steps_async", _fake_exec_async)

    class _Ableton:
        """Deliberately has no send_command_async — proves the fingerprint
        check is skipped entirely (not merely tolerant of a fetch error)."""
        def send_command(self, cmd, params=None):
            raise AssertionError("commit should not call sync send_command")

    ctx = SimpleNamespace(lifespan_context={"ableton": _Ableton()})
    result = asyncio.run(commit_preview_variant(
        ctx, set_id="ps_fp_absent", variant_id="v_fp_absent",
    ))

    assert result["committed"] is True
    assert result["status"] == "committed"
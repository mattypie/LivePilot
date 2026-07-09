"""Async MCP tools must not run blocking send_command on the event loop.

Three async-def FastMCP tools were doing synchronous TCP round-trips
(``ableton.send_command``) directly in their bodies. Because FastMCP awaits
async tools on the single asyncio loop (sync tools are threadpooled), each
blocking round-trip froze the *entire* server — every other concurrent async
handler, the UDP analyzer endpoint, and bridge I/O — for up to RECV_TIMEOUT.

Tools covered:
  * transport.get_session_diagnostics  (mcp_server/tools/transport.py)
  * semantic_moves.apply_semantic_move  (mcp_server/semantic_moves/tools.py)
  * mixing.get_track_meters             (mcp_server/tools/mixing.py)

The fix wraps each pre-await ``send_command`` in ``await asyncio.to_thread(...)``.

Each test asserts two things:
  1. Behavior is unchanged — the fake ableton records call commands/params in
     order and the tool still returns the correct merged result.
  2. The call is actually offloaded — a fake ``send_command`` that *blocks on a
     threading.Event* is only released by a *concurrent coroutine* scheduled on
     the same loop. If the tool ran send_command on the loop, that releaser
     could never run and the test would dead­lock (caught by asyncio.wait_for).
     This is the discriminating check: it passes only when send_command runs
     off-loop via to_thread.
"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest


# ── Fakes ──────────────────────────────────────────────────────────────────


class RecordingAbleton:
    """Records each send_command call (command, params) in order and returns a
    canned response keyed by command."""

    def __init__(self, responses: dict):
        self.responses = responses
        self.calls: list[tuple] = []

    def send_command(self, command, params=None):
        self.calls.append((command, params))
        resp = self.responses.get(command)
        return dict(resp) if isinstance(resp, dict) else resp

    # transport.get_recent_actions path uses this; harmless to expose.
    def get_recent_commands(self, limit):
        return []


class BlockingAbleton:
    """send_command blocks on a threading.Event until release() is called.

    If send_command were invoked directly on the event loop, the loop could
    never reach the concurrent coroutine that calls release() — so the only way
    these tests complete is if send_command runs off-loop (asyncio.to_thread)."""

    def __init__(self, responses: dict):
        self.responses = responses
        self.calls: list[tuple] = []
        self._gate = threading.Event()

    def release(self):
        self._gate.set()

    def send_command(self, command, params=None):
        self.calls.append((command, params))
        # Block until a concurrent coroutine releases the gate. Bounded so a
        # genuine regression fails fast instead of hanging the suite forever.
        if not self._gate.wait(timeout=5.0):
            raise AssertionError(
                f"send_command({command!r}) was never released — it appears to "
                "run on the event loop (releaser coroutine could not progress)"
            )
        return dict(self.responses.get(command)) if isinstance(
            self.responses.get(command), dict
        ) else self.responses.get(command)


def _ctx(ableton):
    return SimpleNamespace(lifespan_context={"ableton": ableton})


# ── transport.get_session_diagnostics ──────────────────────────────────────


class TestGetSessionDiagnostics:
    def test_returns_diagnostics_result_unchanged(self):
        from mcp_server.tools.transport import get_session_diagnostics

        ableton = RecordingAbleton(
            {"get_session_diagnostics": {"issues": [], "stats": {"track_count": 3}}}
        )
        result = asyncio.run(get_session_diagnostics(_ctx(ableton)))
        assert result == {"issues": [], "stats": {"track_count": 3}}
        assert ableton.calls == [("get_session_diagnostics", None)]

    def test_send_command_is_offloaded_off_the_event_loop(self):
        from mcp_server.tools.transport import get_session_diagnostics

        ableton = BlockingAbleton(
            {"get_session_diagnostics": {"issues": []}}
        )

        async def driver():
            tool = asyncio.ensure_future(get_session_diagnostics(_ctx(ableton)))
            # Yield so the tool starts and enters the blocking send_command in a
            # worker thread. If it ran on the loop, this releaser never executes.
            await asyncio.sleep(0.05)
            ableton.release()
            return await asyncio.wait_for(tool, timeout=5.0)

        result = asyncio.run(driver())
        assert result == {"issues": []}


# ── semantic_moves.apply_semantic_move ─────────────────────────────────────


class TestApplySemanticMove:
    def test_send_command_is_offloaded_off_the_event_loop(self):
        from mcp_server.semantic_moves.tools import apply_semantic_move
        from mcp_server.semantic_moves.models import SemanticMove
        from mcp_server.semantic_moves.registry import register, _REGISTRY
        from mcp_server.semantic_moves import compiler as move_compiler
        from mcp_server.semantic_moves.compiler import CompiledPlan, CompiledStep

        move_id = "_offload_probe"
        register(SemanticMove(move_id=move_id, family="mix", intent="offload probe"))

        captured: list[dict] = []

        def _probe_compiler(move, kernel):
            captured.append(kernel)
            return CompiledPlan(
                move_id=move.move_id,
                intent=move.intent,
                steps=[CompiledStep(tool="get_session_info", params={},
                                    description="x", backend="remote_command")],
                summary="probe",
            )

        move_compiler.register_compiler(move_id, _probe_compiler)
        try:
            ableton = BlockingAbleton(
                {"get_session_info": {"tempo": 120, "tracks": [], "scenes": []}}
            )

            async def driver():
                # improve mode → compiles but never executes; the ONLY
                # send_command is the pre-await get_session_info we offloaded.
                tool = asyncio.ensure_future(
                    apply_semantic_move(_ctx(ableton), move_id=move_id, mode="improve")
                )
                await asyncio.sleep(0.05)
                ableton.release()
                return await asyncio.wait_for(tool, timeout=5.0)

            result = asyncio.run(driver())
            assert result.get("executed") is False
            # session_info actually flowed into the compiler kernel unchanged.
            assert captured and captured[-1]["session_info"] == {
                "tempo": 120, "tracks": [], "scenes": [],
            }
        finally:
            _REGISTRY.pop(move_id, None)
            move_compiler._COMPILERS.pop(move_id, None)


# ── mixing.get_track_meters ────────────────────────────────────────────────


class TestGetTrackMeters:
    def test_single_sample_results_and_ordering_unchanged(self):
        from mcp_server.tools.mixing import get_track_meters

        ableton = RecordingAbleton({
            "get_track_meters": {"tracks": [{"index": 0, "level": 0.4}]},
            "get_session_info": {"is_playing": True},
        })
        result = asyncio.run(get_track_meters(_ctx(ableton)))
        assert result["tracks"] == [{"index": 0, "level": 0.4}]
        assert result["is_playing"] is True
        # Ordering preserved: meters first, then the session probe.
        assert ableton.calls == [
            ("get_track_meters", {}),
            ("get_session_info", {}),
        ]

    def test_multi_sample_peak_merge_and_call_count_unchanged(self):
        from mcp_server.tools.mixing import get_track_meters

        # Two snapshots; peak-merge should keep the louder level per track.
        responses = iter([
            {"tracks": [{"index": 0, "level": 0.3}]},
            {"tracks": [{"index": 0, "level": 0.7}]},
        ])

        class SeqAbleton(RecordingAbleton):
            def send_command(self, command, params=None):
                self.calls.append((command, params))
                if command == "get_track_meters":
                    return next(responses)
                return {"is_playing": False}

        ableton = SeqAbleton({})
        result = asyncio.run(
            get_track_meters(_ctx(ableton), samples=2, sample_interval_ms=0)
        )
        assert result["tracks"] == [{"index": 0, "level": 0.7}]
        assert result["samples_collected"] == 2
        # 2 meter reads in the loop + 1 session probe, in order.
        meter_calls = [c for c in ableton.calls if c[0] == "get_track_meters"]
        assert len(meter_calls) == 2
        assert ableton.calls[-1][0] == "get_session_info"

    def test_send_command_is_offloaded_off_the_event_loop(self):
        from mcp_server.tools.mixing import get_track_meters

        ableton = BlockingAbleton({
            "get_track_meters": {"tracks": [{"index": 0, "level": 0.5}]},
            "get_session_info": {"is_playing": True},
        })

        async def driver():
            tool = asyncio.ensure_future(get_track_meters(_ctx(ableton)))
            await asyncio.sleep(0.05)
            ableton.release()  # releases the gate for ALL blocked send_commands
            return await asyncio.wait_for(tool, timeout=5.0)

        result = asyncio.run(driver())
        assert result["tracks"] == [{"index": 0, "level": 0.5}]
        assert result["is_playing"] is True


# ── analyzer.compressor_set_sidechain (2026-06-24 ultrareview sweep) ─────────


class TestCompressorSetSidechain:
    def test_returns_result_unchanged(self):
        from mcp_server.tools.analyzer import compressor_set_sidechain

        ableton = RecordingAbleton(
            {"set_compressor_sidechain": {"ok": True, "enabled": True}}
        )
        result = asyncio.run(
            compressor_set_sidechain(
                _ctx(ableton), track_index=0, device_index=1,
                source_type="1-Kick", source_channel="Post FX",
            )
        )
        assert result == {"ok": True, "enabled": True}
        assert ableton.calls == [(
            "set_compressor_sidechain",
            {"track_index": 0, "device_index": 1,
             "source_type": "1-Kick", "source_channel": "Post FX"},
        )]

    def test_send_command_is_offloaded_off_the_event_loop(self):
        from mcp_server.tools.analyzer import compressor_set_sidechain

        ableton = BlockingAbleton({"set_compressor_sidechain": {"ok": True}})

        async def driver():
            tool = asyncio.ensure_future(
                compressor_set_sidechain(_ctx(ableton), track_index=0, device_index=0)
            )
            await asyncio.sleep(0.05)
            ableton.release()
            return await asyncio.wait_for(tool, timeout=5.0)

        assert asyncio.run(driver()) == {"ok": True}


# ── Whole-file guard: no async tool may block on ableton.send_command ───────
#
# Historically this guard only scanned mcp_server/tools/analyzer.py for direct
# `ableton.send_command(...)` calls. The 2026-07 event-loop sweep found (and
# fixed) the same bug class scattered across ~20 files in mcp_server/ — some
# direct, some one hop away through a same-file sync helper — plus adjacent
# blocking primitives (file I/O, subprocess). The generalized guard below
# reuses the transitive AST scanner built for that sweep
# (`scripts/scan_async_blocking.py`) so the whole tree is covered going
# forward, not just the one file that happened to regress before.


def _format_violations(violations) -> str:
    lines = []
    for v in violations:
        if v.kind == "direct":
            lines.append(
                f"{v.file}:{v.lineno}: async def {v.func_name}() -> "
                f"direct unwrapped {v.detail}"
            )
        else:
            lines.append(
                f"{v.file}:{v.lineno}: async def {v.func_name}() -> calls helper "
                f"{v.helper_name}() (line {v.helper_lineno}) which has unwrapped "
                f"{v.detail}"
            )
    return "\n".join(lines)


def test_analyzer_async_tools_never_call_send_command_on_the_loop():
    """Regression guard for the 2026-06-24 ultrareview analyzer.py sweep,
    subsumed by (and kept passing under) the general tree-wide guard below.

    Every `async def` in mcp_server/tools/analyzer.py must offload the
    synchronous TCP client — direct `ableton.send_command(...)` calls block
    the single asyncio event loop for the whole round-trip (up to
    RECV_TIMEOUT), freezing every concurrent handler + the analyzer bridge.
    """
    from pathlib import Path

    from scripts.scan_async_blocking import scan_file

    analyzer = (
        Path(__file__).resolve().parents[1] / "mcp_server" / "tools" / "analyzer.py"
    )
    violations = scan_file(analyzer)
    assert not violations, (
        "async analyzer tools must offload blocking calls off the event loop "
        f"(via asyncio.to_thread / send_command_async). Offenders:\n"
        f"{_format_violations(violations)}"
    )


def test_all_mcp_server_async_tools_never_block_the_event_loop():
    """Tree-wide regression guard for the "blocking call on the event loop"
    bug class across ALL of mcp_server/.

    Walks every `.py` file under `mcp_server/` (skipping `__pycache__` and
    `mcp_server/splice_client/protos`, which are generated/vendored) and
    flags, inside every `async def`:

      * direct `ableton.send_command(...)` / `<conn>.send_command(...)` calls
        where the receiver reads as an AbletonConnection (ableton/conn/
        connection — NOT bridge/m4l/spectral, which are already async-native
        and must never be wrapped in `to_thread`),
      * `Path.write_bytes` / `write_text` / `read_bytes` / `read_text` /
        `mkdir` file I/O,
      * `subprocess.run` / `call` / `check_call` / `check_output` / `Popen`,

    unless the call is passed as a *reference* into `asyncio.to_thread(...)`
    or `loop.run_in_executor(...)`, or the wrapper `send_command_async(...)`
    is used directly. Also catches the one-hop case: a plain `def` helper
    containing a flagged call, invoked bare (not through `to_thread`) from an
    `async def` in the same file.

    A hit here means some concurrent MCP tool call — and the UDP analyzer
    bridge — can freeze for the duration of a blocking round-trip. See
    `scripts/scan_async_blocking.py` for the full scanner implementation
    (reused here, not reimplemented) and its module docstring for the
    exemption rules in detail.
    """
    from pathlib import Path

    from scripts.scan_async_blocking import scan_file

    repo_root = Path(__file__).resolve().parents[1]
    mcp_server_root = repo_root / "mcp_server"

    def _is_skipped(path: Path) -> bool:
        parts = path.relative_to(mcp_server_root).parts
        return "__pycache__" in parts or parts[: len(("splice_client", "protos"))] == (
            "splice_client",
            "protos",
        )

    violations: list = []
    for path in sorted(mcp_server_root.rglob("*.py")):
        if _is_skipped(path):
            continue
        violations.extend(scan_file(path))

    assert not violations, (
        "found blocking call(s) reachable from an `async def` in mcp_server/ "
        "that are not offloaded off the event loop. Fix with "
        "`await <ableton>.send_command_async(...)` or "
        "`await asyncio.to_thread(...)`. Offenders:\n"
        f"{_format_violations(violations)}"
    )

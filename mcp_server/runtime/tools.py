"""MCP tool wrappers for runtime capability state and session kernel.

Tools:
  get_capability_state — probe session + analyzer + memory, return snapshot
  get_session_kernel — build the unified V2 turn snapshot for orchestration
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import urllib.request
from typing import Any, Optional

from fastmcp import Context

from ..server import mcp
from ..memory.technique_store import TechniqueStore
from .capability_state import build_capability_state

logger = logging.getLogger(__name__)

_memory_store = TechniqueStore()
_FLUCOMA_STREAM_KEYS = (
    "spectral_shape",
    "mel_bands",
    "chroma",
    "onset",
    "novelty",
    "loudness",
)


# ── Capability probes ──────────────────────────────────────────────────
#
# These helpers are module-level so tests can monkeypatch them directly.


def _probe_web(timeout: float = 0.5) -> bool:
    """Server-side outbound HTTP probe.

    True when the MCP host can reach an arbitrary public URL. Does NOT
    imply curated research corpora are installed — see the ``research``
    domain for that.

    Implementation: a ``timeout``-second HEAD request to
    ``https://api.github.com`` using stdlib ``urllib.request``. Any
    exception (DNS failure, TLS error, socket timeout, proxy block,
    non-2xx response) collapses to False so the probe is safe to call
    from any code path.
    """
    req = urllib.request.Request("https://api.github.com", method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = getattr(resp, "status", None)
            return status is not None and 200 <= status < 400
    except Exception as exc:  # noqa: BLE001 — swallow everything to False
        logger.debug("_probe_web failed: %s", exc)
        return False


def _flucoma_package_dirs() -> list[Path]:
    """Return likely Max package locations for FluCoMa."""
    home = Path.home()
    if os.name == "nt":
        docs = Path(os.environ.get("USERPROFILE", str(home))) / "Documents"
    else:
        docs = home / "Documents"
    return [
        docs / "Max 9" / "Packages" / "FluidCorpusManipulation",
        docs / "Max 8" / "Packages" / "FluidCorpusManipulation",
    ]


def _probe_flucoma_package() -> bool:
    """Check whether the FluCoMa Max package exists on disk.

    LivePilot's real-time FluCoMa support is Max-for-Live based. There is
    no required Python package named ``flucoma``; probing importability
    caused false ``flucoma_not_installed`` reports on healthy systems.
    """
    try:
        for package_dir in _flucoma_package_dirs():
            if not package_dir.exists():
                continue
            if (package_dir / "package-info.json").exists():
                return True
            externals = package_dir / "externals"
            if externals.exists() and any(externals.glob("fluid.*")):
                return True
        return False
    except Exception as exc:  # noqa: BLE001
        logger.debug("_probe_flucoma_package failed: %s", exc)
        return False


def _probe_flucoma(spectral=None) -> dict:
    """Probe the Max/bridge-backed FluCoMa runtime.

    ``available`` means at least one FluCoMa stream has reached the
    spectral cache. ``device_loaded`` means the external package appears
    installed (or streams prove it is working from a frozen analyzer).
    """
    package_installed = _probe_flucoma_package()
    bridge_connected = bool(
        spectral is not None and getattr(spectral, "is_connected", False)
    )

    streams: dict[str, bool] = {}
    if bridge_connected:
        for key in _FLUCOMA_STREAM_KEYS:
            try:
                streams[key] = spectral.get(key) is not None
            except Exception as exc:  # noqa: BLE001
                logger.debug("_probe_flucoma stream %s failed: %s", key, exc)
                streams[key] = False

    active_streams = sum(1 for present in streams.values() if present)
    available = active_streams > 0
    device_loaded = package_installed or available

    reasons: list[str] = []
    if not device_loaded:
        reasons.append("flucoma_not_installed")
    if not bridge_connected:
        reasons.append("flucoma_bridge_unavailable")
    elif not available:
        reasons.append("flucoma_no_streams")

    return {
        "available": available,
        "device_loaded": device_loaded,
        "bridge_connected": bridge_connected,
        "active_streams": active_streams,
        "streams": streams,
        "reasons": reasons,
    }


def _normalize_flucoma_probe(raw) -> dict:
    """Back-compat shim for tests/extensions that monkeypatch a bool probe."""
    if isinstance(raw, bool):
        return {
            "available": raw,
            "device_loaded": raw,
            "reasons": [] if raw else ["flucoma_not_installed"],
        }
    if isinstance(raw, dict):
        available = bool(raw.get("available", False))
        return {
            **raw,
            "available": available,
            "device_loaded": bool(raw.get("device_loaded", available)),
            "reasons": list(raw.get("reasons", [])),
        }
    return {
        "available": False,
        "device_loaded": False,
        "reasons": ["flucoma_probe_failed"],
    }


def _run_flucoma_probe(spectral=None) -> dict:
    try:
        return _normalize_flucoma_probe(_probe_flucoma(spectral))
    except TypeError:
        # Older tests monkeypatch _probe_flucoma as a no-arg callable.
        return _normalize_flucoma_probe(_probe_flucoma())


def _session_info(ableton) -> dict:
    try:
        info = ableton.send_command("get_session_info")
        return info if isinstance(info, dict) else {}
    except Exception as exc:  # noqa: BLE001
        logger.debug("_session_info probe failed: %s", exc)
        return {}


def _live_version_from_session(session_info: dict) -> str:
    version = session_info.get("live_version")
    return str(version) if version else "12.0.0"


def _normalize_probe_surface(
    raw: Any,
    *,
    default_reason: str,
    default_mode: str = "manual_only",
) -> dict:
    if isinstance(raw, dict):
        mode = str(raw.get("mode") or default_mode)
        available = bool(raw.get("available", mode in {"readable", "routable", "callable"}))
        reasons = list(raw.get("reasons") or [])
        if not available and not reasons:
            reasons.append(default_reason)
        return {
            "mode": mode,
            "available": available,
            "reasons": reasons,
            "observed": dict(raw.get("observed") or {}),
        }
    return {
        "mode": default_mode,
        "available": False,
        "reasons": [default_reason],
        "observed": {},
    }


def _probe_link_audio_surface(ableton, session_info: Optional[dict] = None) -> dict:
    """Read-only Link Audio surface probe.

    Current Live 12.4 builds expose Link Audio as product UX, but no stable
    Remote Script/M4L routing contract has been proven. This helper looks for
    explicit evidence if future Remote Scripts surface it; otherwise it reports
    manual_only with a reason instead of claiming support from version alone.
    """
    info = session_info or _session_info(ableton)
    observed = {
        "peers_visible": bool(info.get("link_audio_peers") or info.get("link_peers")),
        "inputs_visible": bool(info.get("link_audio_inputs") or info.get("audio_inputs")),
        "routing_visible": bool(info.get("link_audio_routing")),
    }
    if observed["peers_visible"] and observed["inputs_visible"] and observed["routing_visible"]:
        return {
            "mode": "routable",
            "available": True,
            "reasons": [],
            "observed": observed,
        }
    if observed["peers_visible"] or observed["inputs_visible"]:
        return {
            "mode": "readable",
            "available": True,
            "reasons": [],
            "observed": observed,
        }
    return {
        "mode": "manual_only",
        "available": False,
        "reasons": ["link_audio_not_exposed"],
        "observed": observed,
    }


def _probe_stem_workflow_surface(ableton, session_info: Optional[dict] = None) -> dict:
    """Read-only selected-time stem workflow probe.

    This intentionally does not invoke stem separation. It only reports
    callable evidence if a future Remote Script/M4L surface advertises a
    stable command path.
    """
    info = session_info or _session_info(ableton)
    callable_paths = list(info.get("stem_callable_paths") or [])
    if callable_paths:
        return {
            "mode": "callable",
            "available": True,
            "reasons": [],
            "observed": {"callable_paths": callable_paths},
        }
    return {
        "mode": "manual_only",
        "available": False,
        "reasons": ["stem_command_not_observable"],
        "observed": {"callable_paths": []},
    }


def _probe_live_12_4_domain(
    *,
    ableton,
    session_info: dict,
    feature_available: bool,
    surface_probe,
    default_reason: str,
) -> dict:
    live_version = _live_version_from_session(session_info)
    if not feature_available:
        return {
            "ok": True,
            "live_version": live_version,
            "mode": "unavailable",
            "available": False,
            "reasons": ["live_version_below_12_4"],
            "observed": {},
        }
    try:
        surface = _normalize_probe_surface(
            surface_probe(ableton, session_info=session_info),
            default_reason=default_reason,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Live 12.4 surface probe failed: %s", exc)
        surface = {
            "mode": "manual_only",
            "available": False,
            "reasons": [f"probe_failed: {type(exc).__name__}"],
            "observed": {},
        }
    return {
        "ok": True,
        "live_version": live_version,
        **surface,
    }


def _probe_link_audio_domain(ableton, session_info: dict) -> dict:
    from .live_version import LiveVersionCapabilities

    caps = LiveVersionCapabilities.from_session_info(session_info)
    return _probe_live_12_4_domain(
        ableton=ableton,
        session_info=session_info,
        feature_available=caps.has_link_audio,
        surface_probe=_probe_link_audio_surface,
        default_reason="link_audio_not_exposed",
    )


def _probe_stem_workflow_domain(ableton, session_info: dict) -> dict:
    from .live_version import LiveVersionCapabilities

    caps = LiveVersionCapabilities.from_session_info(session_info)
    return _probe_live_12_4_domain(
        ableton=ableton,
        session_info=session_info,
        feature_available=caps.has_stem_time_selection,
        surface_probe=_probe_stem_workflow_surface,
        default_reason="stem_command_not_observable",
    )


@mcp.tool()
def probe_link_audio(ctx: Context) -> dict:
    """Read-only probe for Live 12.4 Link Audio MCP controllability."""
    ableton = ctx.lifespan_context["ableton"]
    session_info = _session_info(ableton)
    return _probe_link_audio_domain(ableton, session_info)


@mcp.tool()
def probe_stem_workflow(ctx: Context) -> dict:
    """Read-only probe for Live 12.4 selected-time stem workflow support."""
    ableton = ctx.lifespan_context["ableton"]
    session_info = _session_info(ableton)
    return _probe_stem_workflow_domain(ableton, session_info)


@mcp.tool()
def get_capability_state(ctx: Context) -> dict:
    """Probe the runtime and return a capability state snapshot.

    Checks session connectivity, analyzer freshness, memory availability,
    and reports what modes the system can operate in right now.
    """
    ableton = ctx.lifespan_context["ableton"]
    spectral = ctx.lifespan_context.get("spectral")

    # ── Probe session ───────────────────────────────────────────────
    session_info = _session_info(ableton)
    session_ok = bool(session_info) and "error" not in session_info

    # ── Probe analyzer (M4L bridge) ─────────────────────────────────
    analyzer_ok = False
    analyzer_fresh = False
    if spectral is not None:
        analyzer_ok = spectral.is_connected
        if analyzer_ok:
            # Check if we have recent spectrum data
            snap = spectral.get("spectrum")
            analyzer_fresh = snap is not None

    # ── Probe memory (direct TechniqueStore, not TCP) ────────────────
    memory_ok = False
    try:
        _memory_store.list_techniques(limit=1)
        memory_ok = True
    except Exception as exc:
        logger.debug("get_capability_state failed: %s", exc)
        memory_ok = False

    # ── Web — actually probe outbound HTTP egress ───────────────────
    # Scoped to server-side outbound HTTP reachability; does NOT imply
    # a curated research corpus is installed (see ``research`` domain).
    web_ok = _probe_web()

    # ── FluCoMa — Max package + live bridge streams ─────────────────
    flucoma_probe = _run_flucoma_probe(spectral)

    # ── Live 12.4 probe-first surfaces ──────────────────────────────
    link_probe = _probe_link_audio_domain(ableton, session_info)
    stem_probe = _probe_stem_workflow_domain(ableton, session_info)

    state = build_capability_state(
        session_ok=session_ok,
        analyzer_ok=analyzer_ok,
        analyzer_fresh=analyzer_fresh,
        memory_ok=memory_ok,
        web_ok=web_ok,
        flucoma_ok=flucoma_probe["available"],
        flucoma_device_loaded=flucoma_probe["device_loaded"],
        flucoma_reasons=flucoma_probe["reasons"],
        link_audio_mode=link_probe["mode"],
        link_audio_reasons=link_probe["reasons"],
        stem_workflow_mode=stem_probe["mode"],
        stem_workflow_reasons=stem_probe["reasons"],
    )

    return state.to_dict()


@mcp.tool()
def get_session_kernel(
    ctx: Context,
    request_text: str = "",
    mode: str = "improve",
    aggression: float = 0.5,
    freshness: float = 0.5,
    creativity_profile: str = "",
    sacred_elements: Optional[list] = None,
    synth_hints: Optional[dict] = None,
    operation_profile: str = "studio_deep",
) -> dict:
    """Build the unified turn snapshot for V2 orchestration.

    This is the preferred entrypoint for any complex agentic workflow.
    Assembles: session info, capability state, action ledger, taste profile,
    anti-preferences, and session memory into one canonical snapshot.

    Core params:
      mode: observe | improve | explore | finish | diagnose
      aggression: 0.0 (subtle) to 1.0 (bold) — execution boldness.

    Creative controls (PR2 — branch-native migration, optional):
      freshness: 0.0 (don't surprise me) to 1.0 (surprise me). Read by
        producers (Wonder, synthesis_brain, composer) to bias branch
        generation. Distinct from aggression, which is about applying
        a single move boldly; freshness is about how far to roam.
      creativity_profile: shorthand producer philosophy tag. Known values
        include "surgeon" (targeted), "alchemist" (transformative),
        "sculptor" (synthesis-focused). Empty ⇒ producer picks a default.
      sacred_elements: caller-asserted list of sacred elements that
        override or augment what song_brain infers. Shape matches
        song_brain entries: {element_type, description, salience}.
      synth_hints: focus hints for synthesis_brain; shape is open in PR2
        and firms up in PR9. Typical keys: track_indices, device_paths,
        target_timbre, preferred_devices.
      operation_profile: safety/intent posture for this turn. Known values:
        safe_live, studio_deep, arrangement_build, sound_design_deep,
        release_audit.

    Returns: SessionKernel dict with kernel_id, session topology, capabilities,
    memory context, routing hints, and (if provided) creative controls.
    """
    from .session_kernel import build_session_kernel

    ableton = ctx.lifespan_context["ableton"]
    spectral = ctx.lifespan_context.get("spectral")

    # Core: session info + capability state
    session_info = ableton.send_command("get_session_info")
    session_ok = isinstance(session_info, dict) and "error" not in session_info

    analyzer_ok = False
    analyzer_fresh = False
    if spectral is not None:
        analyzer_ok = spectral.is_connected
        if analyzer_ok:
            analyzer_fresh = spectral.get("spectrum") is not None

    # P2#3 (v1.17.3): probe web + flucoma the same way get_capability_state
    # does, and propagate through. Without this the kernel's capability view
    # lies to orchestration planners.
    web_ok = _probe_web()
    flucoma_probe = _run_flucoma_probe(spectral)
    link_probe = _probe_link_audio_domain(ableton, session_info if isinstance(session_info, dict) else {})
    stem_probe = _probe_stem_workflow_domain(ableton, session_info if isinstance(session_info, dict) else {})

    # v1.17.4: probe memory the same way too. Previously memory_ok=True was
    # hardcoded — if the store raised, the kernel still reported memory
    # available. Same truth-gap class as the v1.17.3 web/flucoma fix.
    memory_ok = False
    try:
        _memory_store.list_techniques(limit=1)
        memory_ok = True
    except Exception as exc:
        logger.debug("get_session_kernel memory probe failed: %s", exc)

    state = build_capability_state(
        session_ok=session_ok,
        analyzer_ok=analyzer_ok,
        analyzer_fresh=analyzer_fresh,
        memory_ok=memory_ok,
        web_ok=web_ok,
        flucoma_ok=flucoma_probe["available"],
        flucoma_device_loaded=flucoma_probe["device_loaded"],
        flucoma_reasons=flucoma_probe["reasons"],
        link_audio_mode=link_probe["mode"],
        link_audio_reasons=link_probe["reasons"],
        stem_workflow_mode=stem_probe["mode"],
        stem_workflow_reasons=stem_probe["reasons"],
    )

    # Optional subcomponents — degrade gracefully, but reach into the SAME
    # session-scoped stores the public memory tools read/write via
    # ctx.lifespan_context.setdefault(...). Creating fresh stores here meant
    # users who recorded anti-preferences, session memory, or taste signals
    # through the MCP tools always saw an empty kernel.
    ledger_summary: dict = {}
    taste_graph: dict = {}
    anti_prefs: list = []
    session_mem: list = []
    kernel_warnings: list[str] = []

    # store_purpose: audit_readonly
    # The world-model kernel builder surfaces ledger state (total moves,
    # memory candidates, last_move, recent_moves) as diagnostic data for
    # downstream consumers. Not an anti-repetition reader — it's a
    # kernel-assembly surface; consumers that want recency should either
    # call SessionLedger.get_recent_moves directly (annotated as
    # anti_repetition) or use get_action_ledger_summary.
    try:
        from .action_ledger import SessionLedger
        ledger = ctx.lifespan_context.get("action_ledger")
        if ledger is None:
            ledger = SessionLedger()
            ctx.lifespan_context["action_ledger"] = ledger
        recent = ledger.get_recent_moves(limit=10)
        ledger_summary = {
            "total_moves": len(ledger._entries),
            "memory_candidate_count": len(ledger.get_memory_candidates()),
            "last_move": ledger.get_last_move().to_dict() if ledger.get_last_move() else None,
            "recent_moves": [entry.to_dict() for entry in recent],
        }
    except Exception as e:
        kernel_warnings.append(f"ledger_unavailable: {e}")

    # Taste graph + anti-prefs — share stores via lifespan_context, use the
    # canonical build_taste_graph() so consumers see dimension_weights shape.
    try:
        from ..memory.taste_graph import build_taste_graph
        from ..memory.taste_memory import TasteMemoryStore
        from ..memory.anti_memory import AntiMemoryStore
        from ..persistence.taste_store import PersistentTasteStore
        taste_store = ctx.lifespan_context.setdefault("taste_memory", TasteMemoryStore())
        anti_store = ctx.lifespan_context.setdefault("anti_memory", AntiMemoryStore())
        persistent = ctx.lifespan_context.setdefault("persistent_taste", PersistentTasteStore())
        graph = build_taste_graph(
            taste_store=taste_store,
            anti_store=anti_store,
            persistent_store=persistent,
        )
        taste_graph = graph.to_dict()
        anti_prefs = [p.to_dict() for p in anti_store.get_anti_preferences()]
    except Exception as e:
        kernel_warnings.append(f"taste_graph_unavailable: {e}")

    try:
        from ..memory.session_memory import SessionMemoryStore
        mem_store = ctx.lifespan_context.setdefault("session_memory", SessionMemoryStore())
        session_mem = [entry.to_dict() for entry in mem_store.get_recent(limit=10)]
    except Exception as e:
        kernel_warnings.append(f"session_memory_unavailable: {e}")

    # v1.17.4: state.to_dict() wraps its output as {"capability_state": {...}}
    # because that shape is what the standalone get_capability_state tool
    # returns. When building the session kernel, that wrapper becomes the
    # ugly double-nested kernel["capability_state"]["capability_state"]["domains"]
    # path. Unwrap once here so kernel consumers get
    # kernel["capability_state"]["domains"] directly.
    _cap_dict = state.to_dict()
    _cap_flat = _cap_dict.get("capability_state", _cap_dict)

    kernel = build_session_kernel(
        session_info=session_info,
        capability_state=_cap_flat,
        request_text=request_text,
        mode=mode,
        aggression=aggression,
        ledger_summary=ledger_summary,
        session_memory=session_mem,
        taste_graph=taste_graph,
        anti_preferences=anti_prefs,
        freshness=freshness,
        creativity_profile=creativity_profile,
        operation_profile=operation_profile,
        sacred_elements=sacred_elements,
        synth_hints=synth_hints,
    )

    # Populate routing hints from conductor when request context is available
    if request_text.strip():
        try:
            from ..tools._conductor import classify_request

            plan = classify_request(request_text)
            kernel.recommended_engines = [r.engine for r in plan.routes[:3]]
            kernel.recommended_workflow = plan.workflow_mode
        except Exception as e:
            kernel_warnings.append(f"conductor_routing_unavailable: {e}")

    result_dict = kernel.to_dict()
    if kernel_warnings:
        # Additive — callers can ignore; debug-mode introspection benefits.
        result_dict["warnings"] = kernel_warnings
    return result_dict

"""ComposerEngine — orchestrate prompt → layers → executable plan.

Pure computation engine. Does NOT call MCP tools directly.
Returns compiled plan dicts that the tool layer (tools.py) executes.

Executability contract (Phase 7 rewrite)
----------------------------------------
The returned plan contains only REAL tool calls with concrete params. It
never emits:
  - pseudo-tools like _agent_pick_best_sample or _apply_technique
  - placeholder strings like "{downloaded_path}"
  - invalid sentinels like device_index: -1 or track_index: -1
  - hardcoded clip_slot_index: 0 for tracks with no source clip

Samples are resolved at PLAN time via sample_resolver.resolve_sample_for_layer.
Layers that don't resolve to a concrete local file are dropped from `plan`
but kept in `layers` for descriptive output, and the unresolved role is
named in `warnings`. Processing chains use step_id + $from_step bindings
to bind set_device_parameter.device_index to the actual inserted device
position returned by insert_device.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..prompt_parser import CompositionIntent, parse_prompt
from .layer_planner import LayerSpec, plan_layers, plan_sections
from ..sample_resolver import resolve_sample_for_layer


# ── Result Models ──────────────────────────────────────────────────

@dataclass
class CompositionResult:
    """Result of a full composition run."""

    intent: CompositionIntent = field(default_factory=CompositionIntent)
    layers: list[LayerSpec] = field(default_factory=list)
    sections: list[dict] = field(default_factory=list)
    plan: list[dict] = field(default_factory=list)        # executable steps only
    credits_estimated: int = 0
    dry_run: bool = False
    warnings: list[str] = field(default_factory=list)
    resolved_samples: dict = field(default_factory=dict)  # role -> local_path

    def to_dict(self) -> dict:
        return {
            "intent": self.intent.to_dict(),
            "layer_count": len(self.layers),
            "layers": [l.to_dict() for l in self.layers],
            "sections": self.sections,
            "plan_step_count": len(self.plan),
            "plan": self.plan,
            "credits_estimated": self.credits_estimated,
            "dry_run": self.dry_run,
            "warnings": self.warnings,
            "resolved_samples": self.resolved_samples,
        }


@dataclass
class AugmentResult:
    """Result of an augmentation run."""

    request: str = ""
    intent: CompositionIntent = field(default_factory=CompositionIntent)
    new_layers: list[LayerSpec] = field(default_factory=list)
    plan: list[dict] = field(default_factory=list)
    credits_estimated: int = 0
    warnings: list[str] = field(default_factory=list)
    resolved_samples: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "request": self.request,
            "intent": self.intent.to_dict(),
            "new_layer_count": len(self.new_layers),
            "new_layers": [l.to_dict() for l in self.new_layers],
            "plan_step_count": len(self.plan),
            "plan": self.plan,
            "credits_estimated": self.credits_estimated,
            "warnings": self.warnings,
            "resolved_samples": self.resolved_samples,
        }


# ── Step builders ──────────────────────────────────────────────────

def _step_set_tempo(tempo: int) -> dict:
    return {
        "tool": "set_tempo",
        "params": {"tempo": tempo},
        "description": f"Set tempo to {tempo} BPM",
    }


def _step_create_midi_track(track_index: int, role: str, step_id: str) -> dict:
    return {
        "step_id": step_id,
        "tool": "create_midi_track",
        "params": {"index": track_index},
        "description": f"Create MIDI track for {role}",
        "role": role,
    }


def _step_set_track_name(track_index: int, role: str) -> dict:
    return {
        "tool": "set_track_name",
        "params": {"track_index": track_index, "name": role.title()},
        "description": f"Name track: {role.title()}",
        "role": role,
    }


def _step_load_sample_to_simpler(track_index: int, layer: LayerSpec, file_path: str) -> dict:
    return {
        "tool": "load_sample_to_simpler",
        "params": {"track_index": track_index, "file_path": file_path},
        "description": f"Load sample into Simpler on track {track_index}",
        "backend": "mcp_tool",
        "role": layer.role,
    }


# NOTE: there used to be a _step_suggest_technique helper here that emitted a
# `suggest_sample_technique` step into the executable plan with params
# {"technique_id": layer.technique_id}. This was broken: the real tool's
# signature is (file_path, intent, philosophy, max_suggestions) and takes
# no technique_id param. The step would have failed at runtime with a
# "required file_path missing" error.
#
# Removed in v1.10.3 (Truth Release). Technique suggestions for composer
# layers are now surfaced in the descriptive result output (result.layers[*].
# technique_id) — the agent can call suggest_sample_technique separately
# with the resolved sample path if it wants per-sample recipe advice. The
# executable plan emits only real, validated tool calls.


def _processing_steps_with_binding(
    track_index: int,
    layer: LayerSpec,
    layer_idx: int,
) -> list[dict]:
    """Build insert_device + set_device_parameter pairs using step_id bindings.

    Each insert_device carries a unique step_id like 'layer_0_dev_1'. The
    following set_device_parameter steps bind their device_index param to
    that id via $from_step — the async router resolves it to the real
    device index returned by insert_device at runtime.
    """
    steps: list[dict] = []
    for dev_idx, device in enumerate(layer.processing):
        device_name = device.get("name", "")
        if not device_name:
            continue
        step_id = f"layer_{layer_idx}_dev_{dev_idx}"
        steps.append({
            "step_id": step_id,
            "tool": "insert_device",
            "params": {
                "track_index": track_index,
                "device_name": device_name,
            },
            "description": f"Insert {device_name} on track {track_index}",
            "role": layer.role,
        })
        for param_name, param_value in device.get("params", {}).items():
            steps.append({
                "tool": "set_device_parameter",
                "params": {
                    "track_index": track_index,
                    "device_index": {"$from_step": step_id, "path": "device_index"},
                    "parameter_name": param_name,
                    "value": param_value,
                },
                "description": f"Set {device_name} {param_name} = {param_value}",
                "role": layer.role,
            })
    return steps


def _mix_steps(track_index: int, layer: LayerSpec) -> list[dict]:
    steps: list[dict] = []
    # dB to linear with 0dB -> 0.85 convention (Ableton native scale)
    linear = max(0.0, min(1.0, 10 ** (layer.volume_db / 20.0) * 0.85))
    steps.append({
        "tool": "set_track_volume",
        "params": {"track_index": track_index, "volume": round(linear, 3)},
        "description": f"Set {layer.role} volume to {layer.volume_db}dB ({linear:.3f} linear)",
        "role": layer.role,
    })
    if layer.pan != 0.0:
        steps.append({
            "tool": "set_track_pan",
            "params": {"track_index": track_index, "pan": layer.pan},
            "description": f"Set {layer.role} pan to {layer.pan}",
            "role": layer.role,
        })
    return steps


def _arrangement_steps(
    track_index: int,
    layer: LayerSpec,
    sections: list[dict],
) -> list[dict]:
    """Emit the Session-View source-clip scaffold for a layer.

    For each layer that appears in at least one section, we emit:

      1. create_clip — a 1-bar MIDI clip in session slot 0 (the source)
      2. add_notes   — a single C3 trigger note so Simpler actually sounds

    The trigger-clip approach is intentionally minimal: Simpler in classic
    mode plays the full sample on every note, so a single C3 at bar 0 is
    enough for a playable, auditionable baseline in Session View. The
    suggest_sample_technique step elsewhere in the plan produces a recipe
    the agent can use later to replace the trigger pattern with something
    more musical.

    BUG-FULL-MODE-18 (rescoped): this used to also emit one
    `create_arrangement_clip` per section, tiling the SAME 1-bar/single-note
    source across every section (intro/build/drop/breakdown/outro all play
    the identical loop). That isn't arrangement — it's pattern
    multiplication dressed up as one, and it's worse than not placing
    anything at all because it *looks* like a finished arrangement to a
    caller inspecting `plan`. `ComposerEngine` is a pure-computation engine
    with no per-section creative content generation (no LLM in the loop,
    no per-section note variation) — that capability lives in the
    agent-designed `apply_full_plan_v2` variant-slot system
    (`composer/full/apply.py`), which real callers of `compose_full_apply`
    already use instead of this deterministic engine (its own docstring
    says as much: "Replaces the deterministic engine path that was prone
    to flat single-pattern arrangements (BUG-FULL-MODE-18)").
    `ComposerEngine.compose()` is reachable through a second path though —
    `commit_experiment` → `escalate_composer_branch` — which has no agent
    turn to design real per-section variants. Porting the variant-slot
    system here would require synthesizing per-section musical content
    from nothing, which is out of scope for a mechanical renumbering/
    dead-code fix and would just relocate the same static content into
    more slots without solving the actual "flat" complaint. So: this
    engine now emits ONLY the Session-View scaffold (source clip + track),
    honestly leaving Arrangement placement undone rather than faking it.
    Callers that need real per-section arrangement variation should use
    `compose_full_apply` (or fall back to the branch-hypothesis scaffold
    plan, which never claimed arrangement placement in the first place).
    """
    active_sections = [s for s in sections if s["name"] in layer.sections]
    if not active_sections:
        return []

    steps: list[dict] = []

    # 1. Source session clip — 1 bar = 4 beats at 4/4
    SOURCE_SLOT = 0
    SOURCE_BEATS = 4.0
    steps.append({
        "tool": "create_clip",
        "params": {
            "track_index": track_index,
            "clip_index": SOURCE_SLOT,
            "length": SOURCE_BEATS,
        },
        "description": f"Create 1-bar source clip for {layer.role} (slot {SOURCE_SLOT})",
        "role": layer.role,
    })

    # 2. Single trigger note at beat 0 — C3 (MIDI 60), spanning the full
    #    clip length (BUG-FULL-MODE-6, 2026-05-01).
    #
    #    Earlier comment claimed "Simpler doesn't gate on note-off in classic
    #    mode", but that's wrong: with Simpler's default Ve Mode=None
    #    (standard ADSR), note-off DOES trigger release. A 1-beat note in a
    #    4-beat clip means only the first beat of the loop plays, then 3
    #    beats of silence, then retrigger on the next clip iteration —
    #    audibly choppy/short. Spanning the full clip length keeps the
    #    sample playing continuously through each iteration.
    #
    #    NB: do NOT extend by adding 0.001 to overlap the loop boundary —
    #    Live's clip looping handles edge-of-clip retriggering cleanly when
    #    duration == clip_length.
    steps.append({
        "tool": "add_notes",
        "params": {
            "track_index": track_index,
            "clip_index": SOURCE_SLOT,
            "notes": [{
                "pitch": 60,                # C3
                "start_time": 0.0,
                "duration": SOURCE_BEATS,   # full clip length (4 beats)
                "velocity": 100,
            }],
        },
        "description": f"Add C3 trigger note to {layer.role} source clip",
        "role": layer.role,
    })

    return steps


# ── Engine ─────────────────────────────────────────────────────────

class ComposerEngine:
    """Orchestrates the full composition pipeline.

    Pure computation — returns compiled plan dicts.
    The tool layer (tools.py) handles actual execution.

    Async because sample resolution may download from Splice over gRPC.
    Filesystem-only callers still get near-instant resolution — the resolver
    only awaits when it actually hits the network.
    """

    async def compose(
        self,
        intent: CompositionIntent,
        dry_run: bool = False,
        max_credits: int = 10,
        search_roots: Optional[list] = None,
        splice_client: object = None,
        browser_client: object = None,
    ) -> CompositionResult:
        """Plan a full multi-layer composition from a CompositionIntent.

        Returns a CompositionResult where `plan` contains only executable
        steps. Unresolved layers are kept in `layers` (descriptive) but
        dropped from `plan`, with warnings naming the unresolved roles.

        splice_client is typically `ctx.lifespan_context["splice_client"]`
        from the tool layer. When connected, its catalog is searched after
        filesystem and remote samples are downloaded one credit at a time
        (subject to the hard floor).
        """
        result = CompositionResult(intent=intent, dry_run=dry_run)

        # v1.24: SECTION_TEMPLATES removed per vocabulary-not-form principle (Task 12).
        # plan_layers and plan_sections will raise until Task 14 rewires this to
        # the LLM-creative full-mode flow. DEPRECATED in v1.24.
        layers = plan_layers(intent)
        sections = plan_sections(intent)
        result.layers = layers
        result.sections = sections
        result.credits_estimated = len(layers)

        if result.credits_estimated > max_credits:
            result.warnings.append(
                f"Estimated {result.credits_estimated} credits needed, "
                f"but budget is {max_credits}. Some layers may use "
                f"downloaded samples or browser fallback."
            )

        plan: list[dict] = []

        # Step 1: Tempo
        plan.append(_step_set_tempo(intent.tempo))

        # Step 2: Per-layer build, resolving samples at plan time
        #
        # BUG-FULL-MODE-15: track_index used to be `layer_idx` — the layer's
        # position in the ORIGINAL (pre-drop) layer list. If an earlier layer
        # dropped as unresolved (no continue happened for it, so no track was
        # ever created for that index), later layers still emitted their
        # stale original index — e.g. layers 0,3,4 survive out of 0..4, and
        # `create_midi_track(index=3)` fails because only 1 track exists.
        # `next_track_index` instead counts only ACTUAL track creations, so
        # surviving layers get contiguous indices (0, 1, 2, ...) matching
        # the tracks that really get created.
        next_track_index = 0
        for layer_idx, layer in enumerate(layers):
            file_path, source = await resolve_sample_for_layer(
                layer,
                search_roots=search_roots,
                splice_client=splice_client,
                browser_client=browser_client,
                credit_budget=max_credits,
            )
            if not file_path:
                result.warnings.append(
                    f"Unresolved sample for layer '{layer.role}' "
                    f"(query: {layer.search_query!r}). Dropped from plan."
                )
                continue

            track_index = next_track_index
            next_track_index += 1

            result.resolved_samples[layer.role] = {"path": file_path, "source": source}

            track_step_id = f"layer_{layer_idx}_track"
            plan.append(_step_create_midi_track(track_index, layer.role, track_step_id))
            plan.append(_step_set_track_name(track_index, layer.role))

            plan.append(_step_load_sample_to_simpler(track_index, layer, file_path))

            # technique_id intentionally NOT emitted as an executable step —
            # see note above _step_suggest_technique removal. layer.technique_id
            # is still surfaced in result.layers for descriptive output.

            plan.extend(_processing_steps_with_binding(track_index, layer, layer_idx))
            plan.extend(_mix_steps(track_index, layer))
            plan.extend(_arrangement_steps(track_index, layer, sections))

        # BUG-FULL-MODE-18 (rescoped): this engine only emits a Session-View
        # scaffold per layer (source clip + trigger note) — it does not
        # place anything in Arrangement, since it has no per-section
        # creative content to place. Flag this once so callers (e.g.
        # escalate_composer_branch) don't mistake `plan` for a finished
        # arrangement. Real per-section arrangement variation lives in the
        # agent-designed apply_full_plan_v2 flow (compose_full_apply).
        if layers:
            result.warnings.append(
                "This scaffold places one Session-View source clip per "
                "resolved layer and does not write anything to Arrangement "
                "View — per-section arrangement variation is intentionally "
                "out of scope for this deterministic engine. Use "
                "compose_full_apply's agent-designed variant plan for real "
                "per-section arrangement content."
            )

        result.plan = plan
        return result

    async def augment(
        self,
        request: str,
        max_credits: int = 3,
        max_layers: int = 3,
        search_roots: Optional[list] = None,
        splice_client: object = None,
        browser_client: object = None,
    ) -> AugmentResult:
        """Plan augmentation layers to add to an existing session.

        Like compose(), resolves samples at plan time and drops unresolved
        layers. Since the actual track count isn't known at plan time, this
        uses track_index: -1 only for create_midi_track (where the Remote
        Script interprets -1 as append-at-end) and then binds later steps
        to the actual created track via $from_step — same pattern as the
        device_index binding in compose().
        """
        intent = parse_prompt(request)
        intent.layer_count = min(intent.layer_count or max_layers, max_layers)

        result = AugmentResult(request=request, intent=intent)

        layers = plan_layers(intent)[:max_layers]
        result.new_layers = layers
        result.credits_estimated = len(layers)

        if result.credits_estimated > max_credits:
            result.warnings.append(
                f"Estimated {result.credits_estimated} credits needed, "
                f"but budget is {max_credits}."
            )

        plan: list[dict] = []

        for layer_idx, layer in enumerate(layers):
            file_path, source = await resolve_sample_for_layer(
                layer,
                search_roots=search_roots,
                splice_client=splice_client,
                browser_client=browser_client,
                credit_budget=max_credits,
            )
            if not file_path:
                result.warnings.append(
                    f"Unresolved sample for layer '{layer.role}' "
                    f"(query: {layer.search_query!r}). Dropped from plan."
                )
                continue

            result.resolved_samples[layer.role] = {"path": file_path, "source": source}

            # We don't know the absolute track index yet. create_midi_track's
            # result carries "index" (via Remote Script) — later steps bind
            # track_index to that via $from_step. The composer tools layer
            # passes existing_track_count in via a hint when available.
            track_step_id = f"aug_layer_{layer_idx}_track"
            plan.append({
                "step_id": track_step_id,
                "tool": "create_midi_track",
                "params": {"index": -1},  # append at end — Remote Script convention
                "description": f"Create MIDI track for {layer.role}",
                "role": layer.role,
            })

            track_ref = {"$from_step": track_step_id, "path": "index"}

            plan.append({
                "tool": "set_track_name",
                "params": {"track_index": track_ref, "name": f"+ {layer.role.title()}"},
                "description": f"Name new track: + {layer.role.title()}",
                "role": layer.role,
            })

            plan.append({
                "tool": "load_sample_to_simpler",
                "params": {"track_index": track_ref, "file_path": file_path},
                "description": f"Load sample into Simpler",
                "backend": "mcp_tool",
                "role": layer.role,
            })

            # technique_id intentionally NOT emitted (see compose() above).
            # Surfaced in result.new_layers for descriptive output only.

            for dev_idx, device in enumerate(layer.processing):
                device_name = device.get("name", "")
                if not device_name:
                    continue
                dev_step_id = f"aug_layer_{layer_idx}_dev_{dev_idx}"
                plan.append({
                    "step_id": dev_step_id,
                    "tool": "insert_device",
                    "params": {"track_index": track_ref, "device_name": device_name},
                    "description": f"Insert {device_name}",
                    "role": layer.role,
                })
                for param_name, param_value in device.get("params", {}).items():
                    plan.append({
                        "tool": "set_device_parameter",
                        "params": {
                            "track_index": track_ref,
                            "device_index": {"$from_step": dev_step_id, "path": "device_index"},
                            "parameter_name": param_name,
                            "value": param_value,
                        },
                        "description": f"Set {device_name} {param_name} = {param_value}",
                        "role": layer.role,
                    })

            linear = max(0.0, min(1.0, 10 ** (layer.volume_db / 20.0) * 0.85))
            plan.append({
                "tool": "set_track_volume",
                "params": {"track_index": track_ref, "volume": round(linear, 3)},
                "description": f"Set {layer.role} volume to {layer.volume_db}dB",
                "role": layer.role,
            })
            if layer.pan != 0.0:
                plan.append({
                    "tool": "set_track_pan",
                    "params": {"track_index": track_ref, "pan": layer.pan},
                    "description": f"Set {layer.role} pan to {layer.pan}",
                    "role": layer.role,
                })

        result.plan = plan
        return result

    async def get_plan(
        self,
        intent: CompositionIntent,
        search_roots: Optional[list] = None,
        splice_client: object = None,
        browser_client: object = None,
    ) -> dict:
        """Dry run — return the full composition plan without execution.

        Passes resolution dependencies through to compose() so the dry-run
        accurately reflects which layers would resolve.
        """
        result = await self.compose(
            intent,
            dry_run=True,
            max_credits=0,
            search_roots=search_roots,
            splice_client=splice_client,
            browser_client=browser_client,
        )
        return result.to_dict()

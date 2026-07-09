"""Concrete compilers for mix-domain semantic moves.

Each compiler function takes (move, kernel) and returns a CompiledPlan
with fully parameterized tool calls. The compiler inspects the kernel's
track topology and device chains to resolve targets.

Pure functions — no I/O. All data comes from the kernel dict.
"""

from __future__ import annotations

from .compiler import CompiledPlan, CompiledStep, register_compiler
from .models import SemanticMove
from . import resolvers


def _kernel_track(kernel: dict, track_index: int) -> dict:
    """Return the raw kernel track dict (with its `devices`) for an index.

    resolvers.find_tracks_by_role returns a slimmed dict without `devices`, so
    device resolution must reach back into the raw session_info track to see the
    chain. Returns {} when not found.
    """
    for t in kernel.get("session_info", {}).get("tracks", []) or []:
        if isinstance(t, dict) and t.get("index") == track_index:
            return t
    return {}


def _find_eq_device_index(track: dict) -> int | None:
    """Return the chain index of an EQ device on this track, or None.

    Only inspects device data the kernel already carries — never assumes a
    device that hasn't been confirmed present. Used so frequency-carve writes
    target a resolved device rather than a blind device_index=0 (the
    wrong-device hazard guarded by tests/test_compiler_safety_contract.py).
    """
    for i, dev in enumerate(track.get("devices", []) or []):
        if not isinstance(dev, dict):
            continue
        name = str(dev.get("name", "")).lower()
        class_name = str(dev.get("class_name", "")).lower()
        if "eq" in name or "eq8" in class_name or "eqeight" in class_name:
            return dev.get("index", i)
    return None


def _find_compressor_device_index(track: dict) -> int | None:
    """Return the chain index of a Compressor on this track, or None.

    Same kernel-snapshot-only discipline as _find_eq_device_index — used so the
    sidechain routing targets a resolved compressor rather than a blind index.
    """
    for i, dev in enumerate(track.get("devices", []) or []):
        if not isinstance(dev, dict):
            continue
        name = str(dev.get("name", "")).lower()
        class_name = str(dev.get("class_name", "")).lower()
        if "compressor" in name or "compressor" in class_name:
            return dev.get("index", i)
    return None


def _compile_make_punchier(move: SemanticMove, kernel: dict) -> CompiledPlan:
    """Compile 'make_punchier': push drums, pull pads, tighten master bus."""
    steps = []
    warnings = []
    descriptions = []

    # Find drum track
    drum_tracks = resolvers.find_tracks_by_role(kernel, ["drums", "percussion"])
    if not drum_tracks:
        # Fallback: look for "unknown" role tracks (often drums)
        drum_tracks = resolvers.find_tracks_by_role(kernel, ["unknown"])
    if not drum_tracks:
        warnings.append("No drum track found — skipping drum push")

    # Find pad/texture tracks
    pad_tracks = resolvers.find_tracks_by_role(kernel, ["pad", "fx"])

    # Step 1: Read current state
    steps.append(CompiledStep(
        tool="get_track_meters",
        params={"include_stereo": True},
        description="Read current levels for all tracks",
        verify_after=False,
    ))

    # Step 2: Push drum volume
    for dt in drum_tracks[:1]:  # Only first drum track
        idx = dt["index"]
        steps.append(CompiledStep(
            tool="set_track_volume",
            params={"track_index": idx, "volume": 0.75},
            description=f"Push {dt['name']} (track {idx}) to 0.75 for transient punch",
        ))
        descriptions.append(f"Push {dt['name']} volume to 0.75")

    # Step 3: Pull back pads
    for pt in pad_tracks:
        idx = pt["index"]
        steps.append(CompiledStep(
            tool="set_track_volume",
            params={"track_index": idx, "volume": 0.25},
            description=f"Pull {pt['name']} (track {idx}) to 0.25 for contrast",
        ))
        descriptions.append(f"Pull {pt['name']} volume to 0.25")

    # Step 4: Verify
    steps.append(CompiledStep(
        tool="get_track_meters",
        params={"include_stereo": True},
        description="Verify all tracks still producing audio",
    ))

    return CompiledPlan(
        move_id=move.move_id,
        intent=move.intent,
        steps=steps,
        before_reads=[{"tool": "get_track_meters", "params": {"include_stereo": True}}],
        after_reads=[
            {"tool": "get_track_meters", "params": {"include_stereo": True}},
            {"tool": "get_master_spectrum", "params": {}},
        ],
        risk_level="low",
        summary="; ".join(descriptions) if descriptions else "No changes compiled",
        requires_approval=(kernel.get("mode", "improve") != "explore"),
        warnings=warnings,
    )


def _compile_tighten_low_end(move: SemanticMove, kernel: dict) -> CompiledPlan:
    """Compile 'tighten_low_end': reduce sub volume, boost bass harmonics."""
    steps = []
    warnings = []
    descriptions = []

    bass_tracks = resolvers.find_tracks_by_role(kernel, ["bass"])
    if not bass_tracks:
        warnings.append("No bass track found — cannot tighten low end")
        return CompiledPlan(
            move_id=move.move_id,
            intent=move.intent,
            summary="No bass track found",
            warnings=warnings,
        )

    bass = bass_tracks[0]
    idx = bass["index"]

    # Step 1: Read spectrum (optional — soft-gated on analyzer availability).
    # BUG #3 fix (v1.20.2): if the analyzer isn't loaded on master, this
    # step fails — but the downstream bass-volume change is independent
    # and should still run. optional=True lets the router skip and
    # continue instead of halting the plan.
    steps.append(CompiledStep(
        tool="get_master_spectrum",
        params={},
        description="Read current spectral balance (optional — analyzer-gated)",
        verify_after=False,
        optional=True,
    ))

    # Step 2: Reduce bass volume slightly
    steps.append(CompiledStep(
        tool="set_track_volume",
        params={"track_index": idx, "volume": 0.58},
        description=f"Reduce {bass['name']} volume to 0.58 (tighten sub)",
    ))
    descriptions.append(f"Reduce {bass['name']} volume to 0.58")

    # Step 3: Verify
    steps.append(CompiledStep(
        tool="get_track_meters",
        params={"include_stereo": True},
        description="Verify bass still producing audio after reduction",
    ))

    return CompiledPlan(
        move_id=move.move_id,
        intent=move.intent,
        steps=steps,
        before_reads=[{"tool": "get_master_spectrum", "params": {}}],
        after_reads=[
            {"tool": "get_master_spectrum", "params": {}},
            {"tool": "get_track_meters", "params": {"include_stereo": True}},
        ],
        risk_level="low",
        summary="; ".join(descriptions),
        requires_approval=(kernel.get("mode", "improve") != "explore"),
        warnings=warnings,
    )


def _compile_widen_stereo(move: SemanticMove, kernel: dict) -> CompiledPlan:
    """Compile 'widen_stereo': pan harmonic elements wider, add depth.

    Fallback: when no lead/harmonic tracks are role-classified (e.g., tracks
    are named "Q-Call"/"A-Answer" which don't match role keywords), fall back
    to widening prominent non-drum / non-bass tracks. This prevents the move
    from no-opping on mixes with unusually-named tracks.
    """
    steps = []
    warnings = []
    descriptions = []

    chord_tracks = resolvers.find_tracks_by_role(kernel, ["chords", "pad"])
    lead_tracks = resolvers.find_tracks_by_role(kernel, ["lead"])
    perc_tracks = resolvers.find_tracks_by_role(kernel, ["percussion"])

    # Fallback: if no harmonic/lead tracks found via role, use prominent
    # non-drum/non-bass tracks (any role except drums/bass/fx/unknown=skip).
    # This covers real-world mixes where melodic tracks are named after their
    # musical function ("Q-Call", "A-Answer", "Melody 1") rather than role keywords.
    using_fallback = False
    if not chord_tracks and not lead_tracks:
        all_tracks = resolvers.find_tracks_by_role(
            kernel, ["chords", "pad", "lead", "percussion", "fx", "unknown"]
        )
        # Exclude drums and bass by collecting everything non-drum/non-bass
        exclude_roles = {"drums", "bass"}
        all_raw_tracks = kernel.get("session_info", {}).get("tracks", [])
        fallback_tracks = [
            {
                "index": t.get("index", 0),
                "name": t.get("name", ""),
                "role": resolvers.infer_role(t.get("name", "")),
                "volume": t.get("volume"),
                "pan": t.get("pan"),
                "mute": t.get("mute", False),
                "solo": t.get("solo", False),
            }
            for t in all_raw_tracks
            if resolvers.infer_role(t.get("name", "")) not in exclude_roles
        ]
        if fallback_tracks:
            # Use up to first 2 as a left/right pair
            chord_tracks = fallback_tracks[:1]
            lead_tracks = fallback_tracks[1:2]
            using_fallback = True
            warnings.append(
                "No harmonic or lead tracks found by role — "
                "falling back to widening prominent non-drum/non-bass tracks: "
                + ", ".join(t["name"] for t in fallback_tracks[:2])
            )
        else:
            warnings.append("No harmonic, lead, or wideable tracks found — no changes")

    # Pan chords/first-fallback left
    for ct in chord_tracks[:1]:
        steps.append(CompiledStep(
            tool="set_track_pan",
            params={"track_index": ct["index"], "pan": -0.35},
            description=f"Pan {ct['name']} left (-35%) for width"
            + (" [fallback]" if using_fallback else ""),
        ))
        descriptions.append(f"Pan {ct['name']} left 35%")

    # Pan lead/second-fallback right
    for lt in lead_tracks[:1]:
        steps.append(CompiledStep(
            tool="set_track_pan",
            params={"track_index": lt["index"], "pan": 0.30},
            description=f"Pan {lt['name']} right (+30%) for width"
            + (" [fallback]" if using_fallback else ""),
        ))
        descriptions.append(f"Pan {lt['name']} right 30%")

    # Pan perc slightly (only when using primary role classification)
    if not using_fallback:
        for pt in perc_tracks[:1]:
            steps.append(CompiledStep(
                tool="set_track_pan",
                params={"track_index": pt["index"], "pan": 0.15},
                description=f"Pan {pt['name']} slightly right (+15%)",
            ))
            descriptions.append(f"Pan {pt['name']} slightly right")

    # Verify
    steps.append(CompiledStep(
        tool="get_track_meters",
        params={"include_stereo": True},
        description="Verify stereo output on all panned tracks",
    ))

    return CompiledPlan(
        move_id=move.move_id,
        intent=move.intent,
        steps=steps,
        before_reads=[{"tool": "analyze_mix", "params": {}}],
        after_reads=[
            {"tool": "analyze_mix", "params": {}},
            {"tool": "get_track_meters", "params": {"include_stereo": True}},
        ],
        risk_level="low",
        summary="; ".join(descriptions) if descriptions else "No panning changes",
        requires_approval=(kernel.get("mode", "improve") != "explore"),
        warnings=warnings,
    )


def _compile_darken_mix(move: SemanticMove, kernel: dict) -> CompiledPlan:
    """Compile 'darken_without_losing_width': reduce brightness, preserve stereo."""
    steps = []
    descriptions = []

    # Find bright-sounding tracks (leads and chords)
    bright_tracks = resolvers.find_tracks_by_role(kernel, ["lead", "chords", "percussion"])

    for bt in bright_tracks[:2]:  # Max 2 tracks
        # Reduce volume slightly to darken
        steps.append(CompiledStep(
            tool="set_track_volume",
            params={"track_index": bt["index"], "volume": 0.40},
            description=f"Pull {bt['name']} to 0.40 for darker tone",
        ))
        descriptions.append(f"Darken {bt['name']} to 0.40")

    steps.append(CompiledStep(
        tool="get_track_meters",
        params={"include_stereo": True},
        description="Verify all tracks still active after darkening",
    ))

    return CompiledPlan(
        move_id=move.move_id,
        intent=move.intent,
        steps=steps,
        before_reads=[{"tool": "get_master_spectrum", "params": {}}],
        after_reads=[{"tool": "get_master_spectrum", "params": {}}],
        risk_level="low",
        summary="; ".join(descriptions) if descriptions else "No changes",
        requires_approval=(kernel.get("mode", "improve") != "explore"),
    )


def _compile_reduce_repetition(move: SemanticMove, kernel: dict) -> CompiledPlan:
    """Compile 'reduce_repetition_fatigue': add perlin automation for organic movement."""
    steps = []
    descriptions = []

    # Find melodic tracks for filter drift
    melodic = resolvers.find_tracks_by_role(kernel, ["chords", "lead", "pad"])

    for mt in melodic[:2]:  # Max 2 tracks
        steps.append(CompiledStep(
            tool="apply_automation_shape",
            params={
                "track_index": mt["index"],
                "clip_index": 0,
                "parameter_type": "send",
                "send_index": 0,
                "curve_type": "perlin",
                "center": 0.2,
                "amplitude": 0.1,
                "duration": 8,
                "density": 16,
            },
            description=f"Add perlin reverb send drift on {mt['name']}",
        ))
        descriptions.append(f"Perlin reverb drift on {mt['name']}")

    steps.append(CompiledStep(
        tool="get_track_meters",
        params={"include_stereo": True},
        description="Verify tracks still active after automation",
    ))

    return CompiledPlan(
        move_id=move.move_id,
        intent=move.intent,
        steps=steps,
        before_reads=[],
        after_reads=[{"tool": "get_track_meters", "params": {"include_stereo": True}}],
        risk_level="medium",
        summary="; ".join(descriptions) if descriptions else "No melodic tracks found",
        requires_approval=(kernel.get("mode", "improve") != "explore"),
    )


def _compile_make_kick_bass_lock(move: SemanticMove, kernel: dict) -> CompiledPlan:
    """Compile 'make_kick_bass_lock': carve frequency space between kick and bass.

    Strategy (real frequency separation + timing):
      1. Read bass EQ/filter state first so we know what's already there.
      2. Insert an EQ Eight on the bass track (or use an existing one) and
         apply a high-pass shelf dip in the kick's fundamental band (40-80 Hz)
         to clear sub space for the kick.
      3. Set up a sidechain compressor on the bass keyed from the kick track
         so bass ducks on every kick hit — the defining "lock" gesture.
      4. Optional: minor volume trim only if needed after the above two steps.
    """
    steps: list[CompiledStep] = []
    warnings: list[str] = []
    descriptions: list[str] = []

    bass_tracks = resolvers.find_tracks_by_role(kernel, ["bass"])
    kick_tracks = resolvers.find_tracks_by_role(kernel, ["drums", "percussion"])

    if not bass_tracks:
        warnings.append("No bass track found — cannot lock kick and bass")
    if not kick_tracks:
        warnings.append("No kick/drum track found — reference track missing")

    # Optional pre-read — soft-gated on analyzer availability.
    # BUG #3 fix (v1.20.2): see tighten_low_end for the same pattern.
    steps.append(CompiledStep(
        tool="get_master_spectrum",
        params={},
        description="Read current sub/low balance before carving (optional — analyzer-gated)",
        verify_after=False,
        optional=True,
    ))

    if bass_tracks:
        bass = bass_tracks[0]
        bass_idx = bass["index"]

        # Step: read bass device chain state before touching it
        steps.append(CompiledStep(
            tool="get_device_parameters",
            params={"track_index": bass_idx, "device_index": 0},
            description=f"Read {bass['name']} device chain state before carving",
            verify_after=False,
            optional=True,
        ))

        # Frequency separation: carve a high-pass into an EQ on the bass so the
        # kick's sub-fundamental has room. We only write EQ band parameters when
        # the kernel already exposes a resolvable EQ device on this track — a
        # device inserted in THIS plan has no known index at compile time, so a
        # device_index=0 write would be a blind (wrong-device) target. See
        # tests/test_compiler_safety_contract.py for the invariant this honors.
        eq_idx = _find_eq_device_index(_kernel_track(kernel, bass_idx))
        if eq_idx is not None:
            # EQ confirmed present in the kernel snapshot → safe to carve it.
            # EQ Eight band 1: "1 FilterType" (0 = Low Cut / High-pass),
            # "1 Frequency" (Hz).
            steps.append(CompiledStep(
                tool="set_device_parameter",
                params={
                    "track_index": bass_idx,
                    "device_index": eq_idx,
                    "parameter_name": "1 FilterType",
                    "value": 0.0,  # Low Cut / High-pass
                },
                description=f"Set EQ band 1 to High-Pass on {bass['name']} (device {eq_idx})",
            ))
            steps.append(CompiledStep(
                tool="set_device_parameter",
                params={
                    "track_index": bass_idx,
                    "device_index": eq_idx,
                    "parameter_name": "1 Frequency",
                    "value": 60.0,   # 60 Hz — clears kick sub without losing bass body
                },
                description=f"Set HP cutoff to 60 Hz on {bass['name']} EQ (device {eq_idx})",
            ))
            descriptions.append("HP carve @ 60 Hz on existing EQ")
        else:
            # No resolvable EQ → insert one as a scaffold. The HP-carve must be
            # applied AFTER insertion (the new device's index isn't known until
            # runtime); the sidechain below provides the timing lock regardless.
            steps.append(CompiledStep(
                tool="insert_device",
                params={
                    "track_index": bass_idx,
                    "device_name": "EQ Eight",
                    "position": 0,
                },
                description=f"Insert EQ Eight on {bass['name']} for kick-sub carve",
            ))
            descriptions.append(f"EQ Eight on {bass['name']}")
            warnings.append(
                f"Apply a band-1 high-pass (~60 Hz) to the new EQ on "
                f"{bass['name']} after insertion — device index is not "
                "resolvable at plan time."
            )

        # Timing lock: sidechain a compressor on the bass, keyed from the kick,
        # so the bass ducks on every kick hit. The real compressor_set_sidechain
        # tool only ROUTES the detector input on an EXISTING compressor
        # (signature: track_index, device_index, source_type, source_channel —
        # it does NOT set threshold/ratio/attack/release). So resolve a
        # compressor from the kernel snapshot or insert one as a scaffold (its
        # post-insert index isn't known at plan time → no blind device writes).
        if kick_tracks:
            kick = kick_tracks[0]
            comp_idx = _find_compressor_device_index(_kernel_track(kernel, bass_idx))
            if comp_idx is not None:
                steps.append(CompiledStep(
                    tool="set_device_parameter",
                    params={
                        "track_index": bass_idx,
                        "device_index": comp_idx,
                        "parameter_name": "S/C On",
                        "value": 1.0,
                    },
                    description=f"Enable sidechain on {bass['name']} compressor (device {comp_idx})",
                ))
                steps.append(CompiledStep(
                    tool="compressor_set_sidechain",
                    params={
                        "track_index": bass_idx,
                        "device_index": comp_idx,
                        "source_type": kick["name"],
                        "source_channel": "Post FX",
                    },
                    description=(
                        f"Route {kick['name']} into {bass['name']} compressor "
                        "sidechain — bass ducks on every kick hit"
                    ),
                ))
                descriptions.append(f"Sidechain {bass['name']} ← {kick['name']}")
            else:
                steps.append(CompiledStep(
                    tool="insert_device",
                    params={
                        "track_index": bass_idx,
                        "device_name": "Compressor",
                    },
                    description=f"Insert Compressor on {bass['name']} for kick sidechain",
                ))
                descriptions.append(f"Compressor on {bass['name']}")
                warnings.append(
                    f"After inserting the Compressor on {bass['name']}, enable "
                    f"'S/C On' and route {kick['name']} (Post FX) into its "
                    "sidechain — device index is not resolvable at plan time."
                )

    steps.append(CompiledStep(
        tool="get_track_meters",
        params={"include_stereo": True},
        description="Verify kick and bass both still producing audio after carve",
    ))

    return CompiledPlan(
        move_id=move.move_id,
        intent=move.intent,
        steps=steps,
        before_reads=[{"tool": "get_master_spectrum", "params": {}}],
        after_reads=[
            {"tool": "get_master_spectrum", "params": {}},
            {"tool": "get_track_meters", "params": {"include_stereo": True}},
        ],
        risk_level="low",
        summary="; ".join(descriptions) if descriptions else "No kick/bass changes compiled",
        requires_approval=(kernel.get("mode", "improve") != "explore"),
        warnings=warnings,
    )


def _compile_create_buildup_tension(move: SemanticMove, kernel: dict) -> CompiledPlan:
    """Compile 'create_buildup_tension': pull harmony back, raise perc energy.

    We apply volume moves as the minimal, reversible tension-builder. Filter
    rises and send ramps belong in an automation recipe — we issue a tension
    gesture template step if the gesture engine is available, otherwise fall
    back to direct volume changes only.
    """
    steps: list[CompiledStep] = []
    warnings: list[str] = []
    descriptions: list[str] = []

    perc_tracks = resolvers.find_tracks_by_role(kernel, ["drums", "percussion"])
    harmony_tracks = resolvers.find_tracks_by_role(kernel, ["chords", "pad"])

    if not perc_tracks and not harmony_tracks:
        warnings.append("No percussion or harmony tracks found — cannot build tension")

    # Raise perc for energy
    for pt in perc_tracks[:1]:
        steps.append(CompiledStep(
            tool="set_track_volume",
            params={"track_index": pt["index"], "volume": 0.78},
            description=f"Push {pt['name']} to 0.78 for rising energy",
        ))
        descriptions.append(f"Push {pt['name']} to 0.78")

    # Pull harmony slightly to amplify perc contrast
    for ht in harmony_tracks[:1]:
        steps.append(CompiledStep(
            tool="set_track_volume",
            params={"track_index": ht["index"], "volume": 0.35},
            description=f"Pull {ht['name']} to 0.35 to create harmonic vacuum before drop",
        ))
        descriptions.append(f"Pull {ht['name']} to 0.35")

    steps.append(CompiledStep(
        tool="get_track_meters",
        params={"include_stereo": True},
        description="Verify tension steps did not silence any track",
    ))

    return CompiledPlan(
        move_id=move.move_id,
        intent=move.intent,
        steps=steps,
        before_reads=[{"tool": "get_emotional_arc", "params": {}}],
        after_reads=[
            {"tool": "get_emotional_arc", "params": {}},
            {"tool": "get_track_meters", "params": {"include_stereo": True}},
        ],
        risk_level="medium",
        summary="; ".join(descriptions) if descriptions else "No tracks to ratchet",
        requires_approval=(kernel.get("mode", "improve") != "explore"),
        warnings=warnings,
    )


def _compile_smooth_scene_handoff(move: SemanticMove, kernel: dict) -> CompiledPlan:
    """Compile 'smooth_scene_handoff': reduce master volume briefly around the handoff.

    Without knowing which two scenes are involved, the compiler can only do a
    conservative energy dip using master volume. A future version should take
    scene indices via kernel.intent_context and apply targeted crossfades.
    """
    steps: list[CompiledStep] = []
    warnings: list[str] = []
    descriptions: list[str] = []

    # Minimal approach — gentle master dip the agent can reverse easily.
    steps.append(CompiledStep(
        tool="get_master_meters",
        params={},
        description="Record current master level for handoff reference",
        verify_after=False,
    ))

    steps.append(CompiledStep(
        tool="set_master_volume",
        params={"volume": 0.78},
        description="Gentle master dip for transition",
    ))
    descriptions.append("Master dip to 0.78")

    steps.append(CompiledStep(
        tool="get_master_meters",
        params={},
        description="Verify master dip applied without clipping",
    ))

    warnings.append(
        "Scene-aware handoff (from_scene/to_scene) not yet compiled — "
        "this is a conservative energy-dip fallback"
    )

    return CompiledPlan(
        move_id=move.move_id,
        intent=move.intent,
        steps=steps,
        before_reads=[{"tool": "get_emotional_arc", "params": {}}],
        after_reads=[{"tool": "get_emotional_arc", "params": {}}],
        risk_level="low",
        summary="; ".join(descriptions),
        requires_approval=(kernel.get("mode", "improve") != "explore"),
        warnings=warnings,
    )


# ── Register all compilers ──────────────────────────────────────────────────

register_compiler("make_punchier", _compile_make_punchier)
register_compiler("tighten_low_end", _compile_tighten_low_end)
register_compiler("widen_stereo", _compile_widen_stereo)
register_compiler("darken_without_losing_width", _compile_darken_mix)
register_compiler("reduce_repetition_fatigue", _compile_reduce_repetition)
register_compiler("make_kick_bass_lock", _compile_make_kick_bass_lock)
register_compiler("create_buildup_tension", _compile_create_buildup_tension)
register_compiler("smooth_scene_handoff", _compile_smooth_scene_handoff)

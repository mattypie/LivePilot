"""Mix Engine critics — detect mix issues from state data.

Six critics: balance, masking, dynamics, stereo, depth, translation.
All pure computation, zero I/O.

Confidence values
------------------
Every issue's ``confidence`` is a BASE value (the original hand-tuned
literal, preserved as the per-issue-type prior) MODULATED by a data-quality
signal actually present in the state being examined — never a decorative
flat constant. Three modulation shapes are used, each documented at its
helper:

  - ``_sample_size_confidence`` — more contributing tracks behind an
    aggregate (e.g. the average volume `anchor_too_weak` compares against)
    means the aggregate is more trustworthy.
  - ``_measurement_confidence`` — real per-track spectral overlap
    (``severity_basis == "spectral_overlap"``) earns a premium over a
    role-pair heuristic prior (no per-track spectrum available).
  - ``_margin_confidence`` / ``_band_interior_confidence`` — how far a
    measurement sits from the classification boundary. Values right at an
    edge are one measurement-noise-width from flipping to the neighboring
    bucket, so they're discounted; values solidly inside the flagged
    region earn the full base confidence.

All modulated values are bounded to [0.3, 0.95] — even a single strong
signal shouldn't claim near-certainty, and even a borderline call keeps a
floor of honesty (it's still a real detection, just a weak one).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

from .models import (
    BalanceState,
    DepthState,
    DynamicsState,
    MaskingMap,
    MixState,
    StereoState,
)


_CONF_FLOOR = 0.3
_CONF_CEIL = 0.95


def _sample_size_confidence(n: int, base: float, full_n: int) -> float:
    """Scale `base` by how many tracks contributed to the aggregate it rests on.

    n=1 -> 70% of base (a lone track's "average" isn't a meaningful
    baseline to compare against). n>=full_n -> 100% of base (a well-
    supported average). Linear ramp in between.
    """
    frac = min(1.0, max(0, n - 1) / max(1, full_n - 1))
    scaled = base * (0.7 + 0.3 * frac)
    return round(max(_CONF_FLOOR, min(_CONF_CEIL, scaled)), 3)


def _measurement_confidence(measured: bool, base: float) -> float:
    """Scale `base` by whether the severity came from a real measurement.

    measured=True  -> per-track spectral overlap was available (this
                       session's actual tracks were compared) -> premium.
    measured=False -> role-pair heuristic prior only (no per-track
                       spectrum) -> discounted, it's a population prior,
                       not a measurement of THIS mix.
    """
    modifier = 1.2 if measured else 0.8
    return round(max(_CONF_FLOOR, min(_CONF_CEIL, base * modifier)), 3)


def _margin_confidence(margin: float, full_margin: float, base: float) -> float:
    """Scale `base` by how far past a decision boundary a measurement sits.

    margin<=0 (right at the boundary) -> 70% of base — the critic could
    flip on measurement noise alone, so confidence is discounted.
    margin>=full_margin -> 100% of base — solidly inside the flagged
    region.
    """
    frac = min(1.0, max(0.0, margin) / full_margin) if full_margin > 0 else 1.0
    scaled = base * (0.7 + 0.3 * frac)
    return round(max(_CONF_FLOOR, min(_CONF_CEIL, scaled)), 3)


def _band_interior_confidence(value: float, low: float, high: float, base: float) -> float:
    """Scale `base` by how centrally `value` sits within a two-sided band.

    Values near either edge of [low, high] are one measurement-noise-width
    from being classified into the neighboring bucket, so they're
    discounted; values near the band's center get the full base value.
    """
    half_width = (high - low) / 2.0
    if half_width <= 0:
        return round(max(_CONF_FLOOR, min(_CONF_CEIL, base)), 3)
    center = (high + low) / 2.0
    distance_from_center = abs(value - center)
    frac = 1.0 - min(1.0, distance_from_center / half_width)
    scaled = base * (0.7 + 0.3 * frac)
    return round(max(_CONF_FLOOR, min(_CONF_CEIL, scaled)), 3)


# ── MixIssue ───────────────────────────────────────────────────────


@dataclass
class MixIssue:
    """A single detected mix issue."""

    issue_type: str = ""
    critic: str = ""
    severity: float = 0.0
    confidence: float = 0.0
    affected_tracks: list[int] = field(default_factory=list)
    evidence: str = ""
    recommended_moves: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ── Balance Critic ──────────────────────────────────────────────────


def run_balance_critic(balance: BalanceState) -> list[MixIssue]:
    """Detect balance problems: anchor too weak, support too loud."""
    issues: list[MixIssue] = []
    active = [t for t in balance.track_states if not t.mute]

    if not active:
        return issues

    # Compute average volume of active tracks
    avg_vol = sum(t.volume for t in active) / len(active)

    # Confidence scales with how many tracks fed the average being compared
    # against — an average of 2 tracks is a much weaker baseline than an
    # average of 8. full_n=6 is where the average is considered well-
    # supported; below that it's discounted toward 70% of the base value.
    n_active = len(active)
    anchor_confidence = _sample_size_confidence(n_active, base=0.7, full_n=6)
    support_confidence = _sample_size_confidence(n_active, base=0.6, full_n=6)

    # Check if anchor tracks are too quiet
    for t in active:
        if t.track_index in balance.anchor_tracks:
            if t.volume < avg_vol * 0.6:
                issues.append(MixIssue(
                    issue_type="anchor_too_weak",
                    critic="balance",
                    severity=min(1.0, (avg_vol - t.volume) / max(avg_vol, 0.01)),
                    confidence=anchor_confidence,
                    affected_tracks=[t.track_index],
                    evidence=(
                        f"Anchor track '{t.name}' (role={t.role}) at volume "
                        f"{t.volume:.2f}, average is {avg_vol:.2f}"
                    ),
                    recommended_moves=["gain_staging"],
                ))

    # Check if non-anchor tracks are too loud
    for t in active:
        if t.track_index not in balance.anchor_tracks:
            if t.volume > avg_vol * 1.5 and t.role not in ("kick", "bass", "vocal", "lead"):
                issues.append(MixIssue(
                    issue_type="support_too_loud",
                    critic="balance",
                    severity=min(1.0, (t.volume - avg_vol) / max(avg_vol, 0.01)),
                    confidence=support_confidence,
                    affected_tracks=[t.track_index],
                    evidence=(
                        f"Support track '{t.name}' (role={t.role}) at volume "
                        f"{t.volume:.2f}, average is {avg_vol:.2f}"
                    ),
                    recommended_moves=["gain_staging"],
                ))

    return issues


# ── Masking Critic ──────────────────────────────────────────────────


def run_masking_critic(masking: MaskingMap) -> list[MixIssue]:
    """Detect frequency collision issues from masking map."""
    issues: list[MixIssue] = []

    for entry in masking.entries:
        if entry.severity >= 0.4:
            issues.append(MixIssue(
                issue_type="frequency_collision",
                critic="masking",
                severity=entry.severity,
                confidence=_measurement_confidence(entry.measured, base=0.6),
                affected_tracks=[entry.track_a, entry.track_b],
                evidence=(
                    f"Tracks {entry.track_a} and {entry.track_b} collide "
                    f"in {entry.overlap_band} band (severity {entry.severity:.2f})"
                ),
                recommended_moves=["eq_correction"],
            ))

    return issues


# ── Dynamics Critic ─────────────────────────────────────────────────


def run_dynamics_critic(
    dynamics: DynamicsState,
    context: Optional[dict] = None,
) -> list[MixIssue]:
    """Detect dynamics problems: over-compression, flat dynamics, low headroom.

    context: optional hint about the session's intended dynamics target.
        {"target_style": "loud_master"} tells the critic this mix is
        deliberately targeting a loud, heavily-limited master — a
        legitimate style choice (not a defect) — so the `over_compressed`
        check is suppressed rather than false-positiving on intentional
        design. Omitting `context` (the default) preserves the original
        behavior byte-for-byte EXCEPT the evidence string, which now
        states the "assumes a dynamic mix target" assumption explicitly
        so a reader doesn't over-trust the flag for a style that
        intentionally lives in the 3-6dB crest band.
    """
    context = context or {}
    target_style = context.get("target_style", "dynamic")
    issues: list[MixIssue] = []

    if dynamics.over_compressed:
        if target_style == "loud_master":
            # Deliberately loud-master styles target exactly this crest
            # band — that's the intended sound, not flat/over-compressed.
            pass
        else:
            issues.append(MixIssue(
                issue_type="over_compressed",
                critic="dynamics",
                severity=min(1.0, max(0.0, (6.0 - dynamics.crest_factor_db) / 6.0)),
                # Confidence is highest at the center of the 3-6dB band and
                # discounted near either edge, where a small measurement
                # wobble could reclassify the mix as healthy or flat instead.
                confidence=_band_interior_confidence(
                    dynamics.crest_factor_db, 3.0, 6.0, base=0.7
                ),
                affected_tracks=[],
                evidence=(
                    f"Crest factor {dynamics.crest_factor_db:.1f} dB — dynamics "
                    f"read as flat/over-compressed. Assumes a dynamic mix target; "
                    f"loud-master styles may intentionally sit at 3-6dB crest — "
                    f"pass context={{'target_style': 'loud_master'}} if that's the intent."
                ),
                recommended_moves=["bus_compression", "transient_shaping"],
            ))

    elif dynamics.crest_factor_db < 3.0 and dynamics.crest_factor_db > 0:
        issues.append(MixIssue(
            issue_type="flat_dynamics",
            critic="dynamics",
            severity=0.8,
            # Confidence scales with how far below the 3.0dB threshold the
            # crest factor sits — deep in flat territory (near 0dB) is a
            # much more confident call than crest sitting just under 3.0.
            confidence=_margin_confidence(
                3.0 - dynamics.crest_factor_db, full_margin=3.0, base=0.8
            ),
            affected_tracks=[],
            evidence=(
                f"Crest factor {dynamics.crest_factor_db:.1f} dB — "
                f"extremely flat, transients are lost"
            ),
            recommended_moves=["transient_shaping", "gain_staging"],
        ))

    if dynamics.headroom is not None and dynamics.headroom < 1.0:
        issues.append(MixIssue(
            issue_type="low_headroom",
            critic="dynamics",
            severity=min(1.0, (1.0 - dynamics.headroom)),
            # Confidence scales with how far below the 1.0dB threshold
            # headroom sits — near-zero headroom is an unambiguous clip
            # risk, while headroom just under 1.0dB is a borderline call.
            confidence=_margin_confidence(
                1.0 - dynamics.headroom, full_margin=1.0, base=0.9
            ),
            affected_tracks=[],
            evidence=f"Only {dynamics.headroom:.1f} dB headroom — clipping risk",
            recommended_moves=["gain_staging"],
        ))

    return issues


# ── Stereo Critic ───────────────────────────────────────────────────


def run_stereo_critic(stereo: StereoState) -> list[MixIssue]:
    """Detect stereo problems: center collapse, overwide."""
    issues: list[MixIssue] = []

    if stereo.mono_risk:
        # mono_risk requires center_strength > 0.85 AND side_activity < 0.05;
        # confidence scales with how far past BOTH thresholds the mix sits —
        # a mix that's barely over 0.85/barely under 0.05 is a much weaker
        # "essentially mono" call than one deep in that territory.
        margin = (stereo.center_strength - 0.85) + (0.05 - stereo.side_activity)
        issues.append(MixIssue(
            issue_type="center_collapse",
            critic="stereo",
            severity=0.6,
            confidence=_margin_confidence(margin, full_margin=0.2, base=0.7),
            affected_tracks=[],
            evidence=(
                f"Center strength {stereo.center_strength:.2f}, "
                f"side activity {stereo.side_activity:.2f} — "
                f"mix is essentially mono"
            ),
            recommended_moves=["width_adjustment"],
        ))

    if stereo.side_activity > 0.7:
        issues.append(MixIssue(
            issue_type="overwide",
            critic="stereo",
            severity=min(1.0, stereo.side_activity - 0.5),
            confidence=_margin_confidence(
                stereo.side_activity - 0.7, full_margin=0.3, base=0.5
            ),
            affected_tracks=[],
            evidence=(
                f"Side activity {stereo.side_activity:.2f} — "
                f"mix may be too wide, center elements could be weak"
            ),
            recommended_moves=["width_adjustment"],
        ))

    return issues


# ── Depth Critic ────────────────────────────────────────────────────


def run_depth_critic(depth: DepthState) -> list[MixIssue]:
    """Detect depth problems: no separation, excessive wash."""
    issues: list[MixIssue] = []

    if depth.depth_separation < 0.05 and depth.wet_dry_ratio > 0.0:
        issues.append(MixIssue(
            issue_type="no_depth_separation",
            critic="depth",
            severity=0.5,
            confidence=_margin_confidence(
                0.05 - depth.depth_separation, full_margin=0.05, base=0.5
            ),
            affected_tracks=[],
            evidence=(
                f"Depth separation {depth.depth_separation:.3f} — "
                f"all tracks at similar depth, no front/back contrast"
            ),
            recommended_moves=["send_rebalance"],
        ))

    if depth.wash_risk:
        issues.append(MixIssue(
            issue_type="excessive_wash",
            critic="depth",
            severity=min(1.0, depth.wet_dry_ratio),
            confidence=_margin_confidence(
                depth.wet_dry_ratio - 0.6, full_margin=0.4, base=0.6
            ),
            affected_tracks=[],
            evidence=(
                f"Wet/dry ratio {depth.wet_dry_ratio:.2f} — "
                f"excessive reverb/delay washing out the mix"
            ),
            recommended_moves=["send_rebalance"],
        ))

    return issues


# ── Translation Critic ──────────────────────────────────────────────


def run_translation_critic(
    dynamics: DynamicsState,
    stereo: StereoState,
) -> list[MixIssue]:
    """Detect translation risks: mono weakness, harshness risk."""
    issues: list[MixIssue] = []

    # Mono weakness: wide mix with weak center. Confidence scales with how
    # far past BOTH thresholds (side_activity > 0.5, center_strength < 0.3)
    # the mix sits — deep in that territory is a much stronger call than
    # barely past either edge.
    if stereo.side_activity > 0.5 and stereo.center_strength < 0.3:
        margin = (stereo.side_activity - 0.5) + (0.3 - stereo.center_strength)
        issues.append(MixIssue(
            issue_type="mono_weakness",
            critic="translation",
            severity=0.7,
            confidence=_margin_confidence(margin, full_margin=0.4, base=0.6),
            affected_tracks=[],
            evidence=(
                f"Side activity {stereo.side_activity:.2f} with center "
                f"strength {stereo.center_strength:.2f} — mono playback "
                f"will lose significant content"
            ),
            recommended_moves=["width_adjustment", "gain_staging"],
        ))

    # Harshness risk: over-compressed + low headroom. Confidence scales with
    # how far below the 3.0dB headroom threshold the mix sits.
    if dynamics.over_compressed and dynamics.headroom is not None and dynamics.headroom < 3.0:
        issues.append(MixIssue(
            issue_type="harshness_risk",
            critic="translation",
            severity=0.6,
            confidence=_margin_confidence(
                3.0 - dynamics.headroom, full_margin=3.0, base=0.5
            ),
            affected_tracks=[],
            evidence=(
                f"Over-compressed (crest {dynamics.crest_factor_db:.1f} dB) "
                f"with only {dynamics.headroom:.1f} dB headroom — "
                f"will sound harsh on smaller speakers"
            ),
            recommended_moves=["gain_staging", "bus_compression"],
        ))

    return issues


# ── Run all critics ─────────────────────────────────────────────────


def run_all_mix_critics(
    mix_state: MixState,
    dynamics_context: Optional[dict] = None,
) -> list[MixIssue]:
    """Run all six critics and aggregate issues.

    dynamics_context: optional hint forwarded to run_dynamics_critic (see
        its docstring) — e.g. {"target_style": "loud_master"}. Omitting it
        (the default) preserves existing behavior for every current caller.
    """
    issues: list[MixIssue] = []
    issues.extend(run_balance_critic(mix_state.balance))
    issues.extend(run_masking_critic(mix_state.masking))
    issues.extend(run_dynamics_critic(mix_state.dynamics, dynamics_context))
    issues.extend(run_stereo_critic(mix_state.stereo))
    issues.extend(run_depth_critic(mix_state.depth))
    issues.extend(run_translation_critic(mix_state.dynamics, mix_state.stereo))
    return issues

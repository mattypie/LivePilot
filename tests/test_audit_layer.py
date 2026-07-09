"""audit_layer report-shape tests.

These tests exercise the pure-computation `checks` module directly. The
@mcp.tool wrapper is integration-tested via the contract suite + live
session; here we verify each check's logic on synthetic inputs.
"""

from __future__ import annotations

from mcp_server.audit import checks


def _drift_factory_params() -> list[dict]:
    """Bare-default Drift factory fingerprint, captured live 2026-05-08."""
    return [
        {"name": "Pitch Mod Amt 1", "value": 0.5},
        {"name": "Pitch Mod Amt 2", "value": 0.5},
        {"name": "Mod Matrix Amt 2", "value": 0.5},
        {"name": "Mod Matrix Amt 3", "value": 0.5},
        {"name": "Vel > Vol", "value": 0.5},
        {"name": "Spread", "value": 0.10},
        {"name": "Strength", "value": 0.05},
        {"name": "Drift", "value": 0.07},
        {"name": "Thickness", "value": 0.0},
        {"name": "LP Mod Amt 1", "value": 0.97},
        {"name": "LP Mod Amt 2", "value": 0.78},
        {"name": "LFO Amt", "value": 1.0},
    ]


# ── §2 Drift factory-fingerprint detection ──────────────────────────


def test_drift_engagement_score_zero_on_factory():
    count, deviated = checks._drift_engagement_score(_drift_factory_params())
    assert count == 0
    assert deviated == []


def test_drift_engagement_score_counts_deviations():
    params = _drift_factory_params()
    # Move two fingerprint params away from factory
    params[0]["value"] = 0.8   # Pitch Mod Amt 1: 0.5 → 0.8 (deviated)
    params[5]["value"] = 0.5   # Spread: 0.10 → 0.5 (deviated)
    count, deviated = checks._drift_engagement_score(params)
    assert count == 2
    assert "Pitch Mod Amt 1" in deviated
    assert "Spread" in deviated


def test_check_params_drift_pad_bare_default_fails():
    devices = [{"class_name": "Drift", "name": "Drift", "parameters": _drift_factory_params()}]
    result = checks.check_params("pad", devices)
    assert result["severity"] == "fail"
    assert result["issues"][0]["code"] == "unprogrammed_instrument"
    assert "ZERO shaping params" in result["issues"][0]["detail"]


def test_check_params_drift_lead_bare_default_fails():
    devices = [{"class_name": "Drift", "name": "Drift", "parameters": _drift_factory_params()}]
    result = checks.check_params("lead", devices)
    assert result["severity"] == "fail"


def test_check_params_drift_bass_bare_default_fails():
    devices = [{"class_name": "Drift", "name": "Drift", "parameters": _drift_factory_params()}]
    result = checks.check_params("bass", devices)
    assert result["severity"] == "fail"


def test_check_params_drift_drum_role_passes_regardless():
    devices = [{"class_name": "Drift", "name": "Drift", "parameters": _drift_factory_params()}]
    result = checks.check_params("kick", devices)
    assert result["severity"] == "pass"
    assert result["evidence"]["suppressed_for_role"] == "kick"


def test_check_params_drift_with_one_deviation_warns_on_pad():
    """One engaged shaping param is below the engagement threshold."""
    params = _drift_factory_params()
    params[0]["value"] = 0.8  # Pitch Mod Amt 1 engaged
    devices = [{"class_name": "Drift", "name": "Drift", "parameters": params}]
    result = checks.check_params("pad", devices)
    assert result["severity"] == "warn"
    assert result["evidence"]["drift_engagement_deviations"] == 1


def test_check_params_drift_with_two_deviations_passes_on_pad():
    """Two engaged shaping params indicates the user has programmed."""
    params = _drift_factory_params()
    params[0]["value"] = 0.7
    params[5]["value"] = 0.5
    devices = [{"class_name": "Drift", "name": "Drift", "parameters": params}]
    result = checks.check_params("pad", devices)
    assert result["severity"] == "pass"
    assert result["evidence"]["drift_engagement_deviations"] == 2


def test_check_params_drift_preset_with_modulation_deviations_passes():
    """A real preset will deviate on multiple fingerprint params."""
    params = _drift_factory_params()
    # Simulate a preset: filter envelope and pitch envelope both engaged + spread + velocity routing
    params[0]["value"] = 0.65   # Pitch Mod Amt 1
    params[4]["value"] = 0.7    # Vel > Vol
    params[5]["value"] = 0.4    # Spread
    devices = [{"class_name": "Drift", "name": "BoC Wash", "parameters": params}]
    result = checks.check_params("pad", devices)
    assert result["severity"] == "pass"
    assert result["evidence"]["drift_engagement_deviations"] == 3



# ── Role inference ───────────────────────────────────────────────────


def test_infer_role_from_name():
    assert checks.infer_role("Kick", []) == "kick"
    assert checks.infer_role("BD 808", []) == "kick"
    assert checks.infer_role("Snare ghost", []) == "snare"
    assert checks.infer_role("Closed HH", []) == "hat"
    assert checks.infer_role("Sub Bass", []) == "bass"
    assert checks.infer_role("Pad Drift", []) == "pad"
    assert checks.infer_role("Atmos drone", []) == "atmos"
    assert checks.infer_role("Lead arp", []) == "lead"


def test_infer_role_falls_back_to_device_class():
    assert checks.infer_role("Track 7", [{"class_name": "DrumGroup"}]) == "perc"
    # Plain instrument -> lead is the conservative fallback
    assert checks.infer_role("Track 8", [{"class_name": "Operator"}]) == "lead"


def test_infer_role_unknown_when_nothing_matches():
    assert checks.infer_role("Group", [{"class_name": "AudioEffectGroupDevice"}]) == "unknown"


# ── §5.1 Timbre ──────────────────────────────────────────────────────


def test_timbre_n_a_when_no_fingerprint():
    result = checks.check_timbre("kick", None)
    assert result["severity"] == "n/a"


def test_timbre_pass_when_role_matches_dominant_band():
    fingerprint = {"bands": {"SUB_LOW": 0.9, "LOW": 0.7, "MID": 0.4, "HIGH": 0.1, "AIR": 0.05,
                              "LOW_MID": 0.3, "PRESENCE": 0.2}}
    result = checks.check_timbre("kick", fingerprint)
    assert result["severity"] == "pass"


def test_timbre_fail_when_role_mismatches():
    # A "kick" sample that's actually presence/air dominant — genuinely wrong
    fingerprint = {"bands": {"SUB_LOW": 0.05, "LOW": 0.1, "LOW_MID": 0.2, "MID": 0.3,
                              "PRESENCE": 0.95, "HIGH": 0.6, "AIR": 0.4}}
    result = checks.check_timbre("kick", fingerprint)
    assert result["severity"] == "fail"
    assert result["issues"]
    assert result["issues"][0]["code"] == "wrong_band_dominance"


def test_timbre_warn_when_secondary_in_expected():
    # Hat with PRESENCE secondary but MID dominant — close but off
    fingerprint = {"bands": {"SUB_LOW": 0.05, "LOW": 0.1, "LOW_MID": 0.2, "MID": 0.95,
                              "PRESENCE": 0.7, "HIGH": 0.3, "AIR": 0.2}}
    result = checks.check_timbre("hat", fingerprint)
    assert result["severity"] == "warn"
    assert result["issues"][0]["code"] == "off_band_dominance"


# ── §5.2 Sequence ────────────────────────────────────────────────────


def test_sequence_n_a_for_audio_track():
    result = checks.check_sequence("vox", [])
    assert result["severity"] == "n/a"


def test_sequence_flags_no_humanization():
    notes = [{"pitch": 60, "velocity": 100, "duration": 0.5} for _ in range(8)]
    result = checks.check_sequence("snare", [notes])
    codes = {i["code"] for i in result["issues"]}
    assert "no_humanization" in codes


def test_sequence_flags_no_ghosts_on_drums():
    # Snare with all hits at 100, no ghosts
    notes = [{"pitch": 38, "velocity": 100 + (i % 3) * 4, "duration": 0.25} for i in range(8)]
    result = checks.check_sequence("snare", [notes])
    codes = {i["code"] for i in result["issues"]}
    assert "no_ghost_notes" in codes


def test_sequence_passes_humanized_drum_with_ghosts():
    # Mix of accents (90-110) and ghosts (30-40)
    notes = []
    for i in range(16):
        v = 95 + (i % 5) * 3 if i % 4 == 0 else 35 + (i % 3) * 4
        notes.append({"pitch": 38, "velocity": v, "duration": 0.25 + (i % 3) * 0.05})
    result = checks.check_sequence("snare", [notes])
    assert result["severity"] in ("pass", "warn")  # humanized + ghosts present


def test_sequence_flags_low_pitch_variety_on_lead():
    notes = [{"pitch": 60, "velocity": 100 + i, "duration": 0.5 + i * 0.05} for i in range(8)]
    result = checks.check_sequence("lead", [notes])
    codes = {i["code"] for i in result["issues"]}
    assert "low_pitch_variety" in codes


# ── §5.3 Stereo ──────────────────────────────────────────────────────


def test_stereo_flags_panned_bass():
    track_info = {"mixer": {"panning": 0.3}}
    result = checks.check_stereo("bass", track_info)
    assert result["severity"] == "warn"
    assert result["issues"][0]["code"] == "panned_bass"


def test_stereo_passes_centered_bass():
    track_info = {"mixer": {"panning": 0.0}}
    result = checks.check_stereo("bass", track_info)
    assert result["severity"] == "pass"


# ── §5.4 Masking ─────────────────────────────────────────────────────


def test_masking_n_a_without_report():
    result = checks.check_masking(3, None)
    assert result["severity"] == "n/a"


def test_masking_filters_for_target_track():
    # Entries use the real MaskingEntry.to_dict() shape: float `severity` and
    # `overlap_band` (NOT a "band" key or a "high"/"warn" string). The worst
    # kick/bass collision has base severity 0.7, which must roll up to FAIL.
    report = {"masking": {"entries": [
        {"track_a": 1, "track_b": 3, "overlap_band": "sub", "severity": 0.7},
        {"track_a": 2, "track_b": 5, "overlap_band": "mid", "severity": 0.3},
    ]}}
    result = checks.check_masking(3, report)
    assert result["severity"] == "fail"  # 0.7 >= 0.65 threshold triggers fail
    assert len(result["issues"]) == 1


def test_masking_severe_collision_fails_and_names_band():
    # Regression: a 0.7-severity collision must be FAIL (not WARN) and the
    # detail must name the real overlap band, not "?". Previously check_masking
    # compared the float severity against the string "high" (dead FAIL branch)
    # and read a nonexistent "band" key (every detail said "band ?").
    report = {"masking": {"entries": [
        {"track_a": 0, "track_b": 4, "overlap_band": "sub", "severity": 0.7},
    ]}}
    result = checks.check_masking(4, report)
    assert result["severity"] == "fail"
    detail = result["issues"][0]["detail"]
    assert "sub" in detail
    assert "band ?" not in detail


def test_masking_mild_collision_warns():
    # A sub-threshold collision (0.3 < 0.65) stays at WARN, not FAIL.
    report = {"masking": {"entries": [
        {"track_a": 1, "track_b": 2, "overlap_band": "mid", "severity": 0.3},
    ]}}
    result = checks.check_masking(1, report)
    assert result["severity"] == "warn"


# ── §5.5 Modulation ──────────────────────────────────────────────────


def test_modulation_fails_pad_with_zero_routings():
    devices = [{
        "class_name": "Drift",
        "parameters": [
            {"name": "Fil < Env", "value": 0.0},
            {"name": "Pe < Env", "value": 0.0},
            {"name": "LFO Amount", "value": 0.0},
        ],
    }]
    result = checks.check_modulation("pad", devices, clip_automation_present=False, wavetable_mod_routings=0)
    assert result["severity"] == "fail"
    assert result["issues"][0]["code"] == "static_layer"


def test_modulation_passes_with_routing():
    devices = [{
        "class_name": "Drift",
        "parameters": [
            {"name": "Fil < Env", "value": 0.5},
        ],
    }]
    result = checks.check_modulation("pad", devices, clip_automation_present=False, wavetable_mod_routings=0)
    assert result["severity"] == "pass"


def test_modulation_passes_with_automation_only():
    devices = [{"class_name": "Drift", "parameters": []}]
    result = checks.check_modulation("pad", devices, clip_automation_present=True, wavetable_mod_routings=0)
    assert result["severity"] == "pass"


# ── §5.6 Params ──────────────────────────────────────────────────────


def test_params_flags_unprogrammed_pad_synth():
    """Bare-default Drift on pad → fail via factory fingerprint."""
    devices = [{
        "class_name": "Drift",
        "parameters": _drift_factory_params(),
    }]
    result = checks.check_params("pad", devices)
    assert result["severity"] == "fail"
    assert result["issues"][0]["code"] == "unprogrammed_instrument"


def test_params_n_a_for_audio_track():
    result = checks.check_params("vox", [{"class_name": "Reverb", "parameters": []}])
    assert result["severity"] == "n/a"


# ── §5.8 Effects ─────────────────────────────────────────────────────


def test_effects_flags_kick_without_eq():
    devices = [{"class_name": "Simpler", "parameters": []}]
    result = checks.check_effects("kick", devices)
    codes = {i["code"] for i in result["issues"]}
    assert "no_eq" in codes


def test_effects_passes_when_chain_is_full():
    devices = [
        {"class_name": "Drift", "parameters": []},
        {"class_name": "EQ Eight", "parameters": []},
        {"class_name": "Compressor", "parameters": []},
        {"class_name": "Reverb", "parameters": []},
    ]
    result = checks.check_effects("lead", devices)
    assert result["severity"] == "pass"


# ── Severity rollup + fix ranking ───────────────────────────────────


def test_rollup_severity_picks_worst():
    check_dict = {
        "a": {"severity": "pass"},
        "b": {"severity": "warn"},
        "c": {"severity": "fail"},
        "d": {"severity": "n/a"},
    }
    assert checks.rollup_severity(check_dict) == "fail"


# ── Live-session shape regressions (validated 2026-05-01) ───────────


def test_multisampler_is_recognized_as_instrument():
    """Phantasm Pad uses class_name='MultiSampler', not 'Sampler'."""
    devices = [{
        "class_name": "MultiSampler",
        "parameters": [
            {"name": "Volume", "value": -16.3},
            {"name": "Filt < Vel", "value": 0.59},
            {"name": "Filt < Key", "value": 1.0},
        ],
    }]
    result = checks.check_params("pad", devices)
    assert result["severity"] != "n/a", "MultiSampler must be recognized as an instrument"


def test_originalsimpler_is_recognized_as_instrument():
    """Hihat 808 Close uses class_name='OriginalSimpler', not 'Simpler'."""
    devices = [{
        "class_name": "OriginalSimpler",
        "parameters": [{"name": "Volume", "value": 0.0}],
    }]
    result = checks.check_samples("hat", devices, slice_classifications=None)
    assert result["severity"] != "n/a", "OriginalSimpler must be recognized as Simpler"


def test_check_samples_tolerates_string_volume_value():
    """Some LOM param shapes serialize `value` as a formatted string
    (e.g. '-12.0 dB'); check_samples must not raise on float()/:.1f — it
    degrades gracefully and skips the numeric default-volume threshold."""
    devices = [{
        "class_name": "Simpler",
        "parameters": [{"name": "Volume", "value": "-12.0 dB"}],
    }]
    result = checks.check_samples("pad", devices, slice_classifications=None)
    assert result["severity"] in ("pass", "warn")
    assert not any(i["code"] == "simpler_default_volume" for i in result["issues"])


def test_modulation_counts_native_velocity_routings():
    """Filt < Vel, Vol < Vel are real routings, not just envelope routings."""
    devices = [{
        "class_name": "MultiSampler",
        "parameters": [
            {"name": "Filt < Vel", "value": 0.59},
            {"name": "Vol < Vel", "value": 0.41},
            {"name": "Filt < Key", "value": 1.0},
            # No envelope amounts and no LFO routings
            {"name": "Fe < Env", "value": 0.0},
            {"name": "Fe On", "value": 0.0},
            {"name": "Filt < LFO", "value": 0.0},
            {"name": "Pe < Env", "value": 0.0},
            {"name": "Pe On", "value": 0.0},
        ],
    }]
    result = checks.check_modulation("pad", devices, clip_automation_present=False, wavetable_mod_routings=0)
    assert result["severity"] == "pass", \
        f"Pad with Filt<Vel + Vol<Vel + Filt<Key should pass §4. Got {result}"
    assert result["evidence"]["routings_count"] >= 2


def test_modulation_does_not_double_count_disabled_envelope():
    """Fe < Env: 0.5 with Fe On: 0 is functionally OFF — should not count."""
    devices = [{
        "class_name": "Drift",
        "parameters": [
            {"name": "Fe < Env", "value": 0.5},
            {"name": "Fe On", "value": 0.0},  # filter env disabled
        ],
    }]
    result = checks.check_modulation("pad", devices, clip_automation_present=False, wavetable_mod_routings=0)
    assert result["severity"] == "fail", \
        "Disabled filter env should not count as a routing"


def test_params_not_flagged_when_filter_env_intentionally_off():
    """Fe < Env: 0 + Fe On: 0 = deliberate, not lazy. Don't flag."""
    devices = [{
        "class_name": "MultiSampler",
        "parameters": [
            {"name": "Fe < Env", "value": 0.0},
            {"name": "Fe On", "value": 0.0},  # explicitly off
            {"name": "Pe < Env", "value": 0.0},
            {"name": "Pe On", "value": 0.0},  # explicitly off
            {"name": "Spread", "value": 30.0},  # programmed
            {"name": "Detune", "value": 5.0},   # programmed
        ],
    }]
    result = checks.check_params("pad", devices)
    # Fe<Env and Pe<Env should be excused. Only counts: nothing problematic.
    assert result["severity"] == "pass"


def test_params_still_flags_when_filter_env_on_but_amount_zero():
    """Fe On: 1 + Fe < Env: 0 IS the lazy case."""
    devices = [{
        "class_name": "Drift",
        "parameters": [
            {"name": "Fe < Env", "value": 0.0},
            {"name": "Fe On", "value": 1.0},  # enabled but no amount
            {"name": "Pe < Env", "value": 0.0},
            {"name": "Pe On", "value": 1.0},  # enabled but no amount
            {"name": "Spread", "value": 0.0},
        ],
    }]
    result = checks.check_params("pad", devices)
    # 3 zero shaping params on a pad = unprogrammed
    assert result["severity"] in ("warn", "fail")


def test_modulation_lfo_routing_requires_lfo_on():
    """Filt < LFO: 0.5 with L On: 0 doesn't actually move anything."""
    devices = [{
        "class_name": "OriginalSimpler",
        "parameters": [
            {"name": "Filt < LFO", "value": 0.5},
            {"name": "L On", "value": 0.0},  # LFO is off
        ],
    }]
    result = checks.check_modulation("pad", devices, clip_automation_present=False, wavetable_mod_routings=0)
    assert result["severity"] == "fail"


def test_modulation_lfo_routing_counts_when_lfo_on():
    devices = [{
        "class_name": "OriginalSimpler",
        "parameters": [
            {"name": "Filt < LFO", "value": 0.5},
            {"name": "L On", "value": 1.0},  # LFO on
        ],
    }]
    result = checks.check_modulation("pad", devices, clip_automation_present=False, wavetable_mod_routings=0)
    assert result["severity"] == "pass"


# ── BUG-E: drum role suppression for many_default_params ─────────────


def test_params_suppresses_default_warn_for_kick():
    """Kick with bare Simpler + default Spread/Detune/Pe<Env/Fe<Env is fine.
    HATS and KICK both falsely warned in 2026-05-01 live test."""
    devices = [{
        "class_name": "OriginalSimpler",
        "parameters": [
            {"name": "Spread", "value": 0.0},
            {"name": "Detune", "value": 0.0},
            {"name": "Pe < Env", "value": 0.0},
            {"name": "Pe On", "value": 1.0},  # envelope on but not used — still fine for kick
            {"name": "Fe < Env", "value": 0.0},
            {"name": "Fe On", "value": 1.0},
        ],
    }]
    result = checks.check_params("kick", devices)
    assert result["severity"] == "pass", \
        f"kick role should suppress many_default_params warn. Got {result}"
    assert result["evidence"].get("suppressed_for_role") == "kick"


def test_params_suppresses_default_warn_for_hat():
    devices = [{
        "class_name": "OriginalSimpler",
        "parameters": [
            {"name": "Spread", "value": 0.0},
            {"name": "Detune", "value": 0.0},
            {"name": "Pe < Env", "value": 0.0},
            {"name": "Pe On", "value": 1.0},
            {"name": "Fe < Env", "value": 0.0},
            {"name": "Fe On", "value": 1.0},
        ],
    }]
    result = checks.check_params("hat", devices)
    assert result["severity"] == "pass"


def test_params_still_strict_for_pad_with_simple_devices():
    """Pad with bare-default Drift fingerprint → still fails (factory fingerprint)."""
    devices = [{
        "class_name": "Drift",
        "parameters": _drift_factory_params(),
    }]
    result = checks.check_params("pad", devices)
    assert result["severity"] == "fail"
    assert result["issues"][0]["code"] == "unprogrammed_instrument"


def test_rank_fixes_orders_by_priority():
    check_dict = {
        "modulation": {"severity": "fail", "issues": [{"code": "static_layer", "detail": "x"}]},
        "effects": {"severity": "warn", "issues": [{"code": "no_eq", "detail": "y"}]},
        "sequence": {"severity": "warn", "issues": [{"code": "uniform_durations", "detail": "z"}]},
    }
    fixes = checks.rank_fixes(check_dict)
    priorities = [f["priority"] for f in fixes]
    # high before medium before low
    assert priorities[0] == "high"
    assert priorities[-1] == "low"


# ── §5.1 timbre check — canonical lowercase bands (review 2026-07-09) ──
#
# Two coupled bugs previously made this check a silent double no-op:
# audit/tools.py called extract_timbre_fingerprint with the wrong signature
# (always "n/a"), and _ROLE_BAND_EXPECTATIONS used uppercase names while
# every spectrum producer emits lowercase (would have flipped to always
# "fail" once the first bug was fixed). These tests pin both fixes.


def _bands(**overrides) -> dict:
    base = {
        "sub_low": 0.02, "sub": 0.03, "low": 0.04, "low_mid": 0.05,
        "mid": 0.06, "high_mid": 0.05, "high": 0.04, "presence": 0.03,
        "air": 0.02,
    }
    base.update(overrides)
    return base


def test_check_timbre_hat_air_presence_passes():
    fp = {"bands": _bands(air=0.6, presence=0.5)}
    result = checks.check_timbre("hat", fp)
    assert result["severity"] == "pass"


def test_check_timbre_kick_sub_dominant_passes():
    fp = {"bands": _bands(sub_low=0.7, sub=0.5)}
    result = checks.check_timbre("kick", fp)
    assert result["severity"] == "pass"


def test_check_timbre_hat_mid_dominant_fails():
    # Mid-dominant "hat" = wrong sample per CLAUDE.md §5.1
    fp = {"bands": _bands(mid=0.8, low_mid=0.7)}
    result = checks.check_timbre("hat", fp)
    assert result["severity"] == "fail"
    assert result["issues"][0]["code"] == "wrong_band_dominance"


def test_check_timbre_secondary_band_in_range_warns():
    # Top band off-role but second band in range → warn, not fail
    fp = {"bands": _bands(low_mid=0.8, presence=0.7)}
    result = checks.check_timbre("hat", fp)
    assert result["severity"] == "warn"


def test_check_timbre_uppercase_producer_still_matches():
    # Defensive case-fold: an uppercase-emitting producer must not fail
    fp = {"bands": {"AIR": 0.6, "PRESENCE": 0.5, "MID": 0.1}}
    result = checks.check_timbre("hat", fp)
    assert result["severity"] == "pass"


def test_maybe_get_timbre_fingerprint_plumbs_cached_spectrum():
    from types import SimpleNamespace
    from mcp_server.audit.tools import _maybe_get_timbre_fingerprint

    class _FakeCache:
        def get(self, key):
            if key == "spectrum":
                return {"value": _bands(air=0.6, presence=0.5)}
            if key == "loudness":
                return {"value": {"rms": 0.2, "peak": 0.6}}
            return None

    ctx = SimpleNamespace(lifespan_context={"spectral": _FakeCache()})
    fp = _maybe_get_timbre_fingerprint(ctx, track_index=3)
    assert fp is not None
    assert fp["source"] == "master_bus_unsoloed"
    assert fp["bands"]["air"] == 0.6
    assert "brightness" in fp["dimensions"]
    # And the fingerprint round-trips into the check
    assert checks.check_timbre("hat", fp)["severity"] == "pass"


def test_maybe_get_timbre_fingerprint_none_without_cache():
    from types import SimpleNamespace
    from mcp_server.audit.tools import _maybe_get_timbre_fingerprint

    ctx = SimpleNamespace(lifespan_context={})
    assert _maybe_get_timbre_fingerprint(ctx, track_index=0) is None

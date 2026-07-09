# Device Parameter Units Reference

Full per-device unit table for `set_device_parameter` / `batch_set_parameters`
(`mcp_server/tools/devices.py`). The tool docstrings carry the most
commonly-hit gotchas inline; this file is the complete list plus the
bug history that motivated the error-enrichment behavior.

## Parameter ranges are NOT always 0-1 (BUG-B4 / B9 / 2026-04-26#2)

Ableton devices use MIXED units depending on the parameter. Always read
the `value_string` in the response (and `min`/`max` from
`get_device_parameters`) before assuming 0-1 semantics:

- Auto Filter `Frequency`: 20-135 index (NOT normalized)
- Auto Filter Legacy `LFO Amount`: 0-30 absolute (displays as %)
- Auto Filter `Resonance`: 0-1.25 on legacy, 0-1 on AutoFilter2
- Auto Filter `Env. Modulation`: -127..+127 on legacy
- Compressor I (legacy): pre-2010 units (Threshold dB direct)
- **Compressor 2 (modern, default)**: 0-1 NORMALIZED. `Threshold 0.85 ≈
  0 dB`, `Ratio 0.75 = 4:1`, `Release 0.16 = 30 ms`. Setting Threshold
  to a dB value like -22 will fail. Compute normalized: `(dB + 50) / 50`
  for typical dB→0-1 mapping, OR read the param's `value_string` after
  a probe write.
- **Saturator** `Drive`, `Output`, `Threshold`, `Color *`: 0-1
  NORMALIZED (Drive 0.5 ≈ 0 dB, Drive 0.6 ≈ +7 dB).
- Dynamic Tube, Vocoder: pre-2010 units
- EQ Three `Frequency Hi/Lo`: 50Hz-15kHz absolute
- Wavetable `Osc 1 Pos`: 0-1 normalized ✓
- Drift / Analog / Operator macros: 0-1 normalized ✓
- Pedal `Output`: -20..+20 dB direct
- Pedal `Bass / Mid / Treble`: -1..+1 direct

## Error enrichment (BUG-2026-04-26#2)

If the Remote Script rejects the value as out-of-range,
`set_device_parameter` fetches the parameter's actual
min/max/value_string and re-raises with that context inline — saves a
follow-up `get_device_parameters` round-trip in the agent loop after
every miss.

## Silent snapping (BUG #4, v1.20.2)

Quantized-enum params (e.g. Beat Repeat's "Gate" at an integer enum)
silently snap a caller's float request to the nearest step. Both
`set_device_parameter` (via the `snapped` field) and
`batch_set_parameters` (via `snapped_params`, tolerance 1e-5) surface
this so callers driving deterministic state don't mistake a snap for
success. `batch_set_parameters` accepts any of `parameter_index`,
`parameter_name`, `name`, `index`, or the legacy `name_or_index` per
entry (BUG-F4 / BUG-2026-04-22#3 — the sibling tools originally had
inconsistent schemas, so all shapes are now normalized so
`get_device_parameters`'s output can be fed straight back in).

## Drum Rack construction workflow (12.3+)

Shared by `insert_device`, `insert_rack_chain`, and the atomic
`add_drum_rack_pad` (see `perception.md` for that one's full history):

1. `insert_device(track_index, 'Drum Rack')` — create empty rack
2. `insert_rack_chain(track_index, device_index)` — add a chain
3. `set_drum_chain_note(chain_index, note=36)` — assign C1 (kick)
4. `insert_device(track_index, 'Simpler', device_index=rack_idx,
   chain_index=0)` — add the instrument into that chain

On Live < 12.3, `insert_device`/`insert_rack_chain` return an error
suggesting `find_and_load_device` instead.

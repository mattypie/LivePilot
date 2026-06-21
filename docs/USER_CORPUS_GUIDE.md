# Building Your Private Corpus-Atlas

How to turn your own Live projects, rack library, Max devices, and plugin presets into queryable AI knowledge — using the same architecture LivePilot uses for its factory pack atlas.

---

## What this is for

The LivePilot factory atlas (`~/.livepilot/atlas-overlays/packs/`) holds 3,917 parsed sidecars across 33 Ableton factory packs — every macro value, every chain, every demo .als track. The `atlas_*` MCP tools query this corpus to answer "what's in this rack", "transplant this aesthetic", "find similar presets".

You probably have *more* useful content on disk than Ableton ships:

- Your own `.als` projects from the last several years — your real workflow
- Your `.adg` rack library — the saved chains you actually use
- Your `.amxd` Max devices — third-party + your own
- Your VST/AU presets — the patches you've made or downloaded

**This guide builds a parallel `~/.livepilot/atlas-overlays/user/` tree** with the same shape as the factory tree, scanned from your content, queryable through the same MCP tools (`extension_atlas_search`, `atlas_*`), composable with AI annotation.

The result: an agent that knows *your* sound, not just Ableton's defaults.

> This guide reflects the v1.27.x corpus tool surface. The corpus is built and
> queried through MCP tools invoked from Claude Code (`corpus_*`,
> `extension_atlas_*`) — there is no standalone `livepilot corpus` shell command.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│   USER FILESYSTEM                                                        │
│  ~/Music/MyProjects/*.als                                                │
│  ~/Music/Ableton/User Library/Presets/**/*.adg                           │
│  ~/Documents/Max 9/Max for Live Devices/**/*.amxd                        │
│  ~/Library/Audio/Presets/**/*.{aupreset,vstpreset,fxp}                   │
└────────────────────┬─────────────────────────────────────────────────────┘
                     │
                     │   corpus_scan  (MCP tool)
                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│   SCANNER REGISTRY (mcp_server/user_corpus/scanners/)                    │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌──────────────┐               │
│   │  als    │  │  adg    │  │  amxd   │  │ plugin-preset│               │
│   └─────────┘  └─────────┘  └─────────┘  └──────────────┘               │
│        ↑           ↑           ↑              ↑                          │
│        └─ register_scanner(cls) — pluggable: write your own              │
└────────────────────┬─────────────────────────────────────────────────────┘
                     │
                     │   per-file Scanner.scan_one(path)
                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│   ~/.livepilot/atlas-overlays/user/                                      │
│   ├── manifest.yaml          # sources, last-scan timestamps, options    │
│   ├── projects/                                                          │
│   │   ├── _parses/*.json     # mechanical sidecar (BPM, devices, ...)    │
│   │   ├── annotated/*.yaml   # OPTIONAL AI annotation (sonic_fingerprint)│
│   │   └── *.yaml             # searchable thin wrappers (entity_id, tags)│
│   ├── racks/                                                             │
│   │   ├── _parses/                                                       │
│   │   ├── annotated/                                                     │
│   │   └── *.yaml                                                         │
│   ├── max_devices/                                                       │
│   └── plugin_presets/                                                    │
└────────────────────┬─────────────────────────────────────────────────────┘
                     │
                     │   overlay loader picks up YAMLs at server boot
                     ▼
┌──────────────────────────────────────────────────────────────────────────┐
│   MCP QUERY SURFACE                                                      │
│   extension_atlas_search(query="warm pad", namespace="user.racks")       │
│   extension_atlas_get(entity_id="user.racks::analog-bass-rumble")        │
│   extension_atlas_list(namespace="user.projects")                        │
│   atlas_macro_fingerprint(... pack_filter=["user.racks"])  # via         │
│                                              # cross-namespace extension │
│   extension_atlas_search("memphis", namespace="user.projects")          │
│   corpus_emit_synthesis_briefs(source_id="my-projects")  # AI annotation │
└──────────────────────────────────────────────────────────────────────────┘
```

**Three layers**:
1. **Mechanical scan** — deterministic, fast, no AI. Produces JSON sidecars + searchable YAML wrappers.
2. **AI annotation** *(opt-in)* — dispatches sonnet subagents to write free-form aesthetic descriptions. Same pattern we used for the 44 factory pack identity YAMLs.
3. **Query surface** — your corpus is reachable from the same MCP tools the factory atlas uses.

---

## File-type coverage

| Content type | Scanner | Extensions | What's captured |
|--------------|---------|------------|-----------------|
| Live projects | `als` | `.als` | BPM, time-sig, scale (with filename-fallback), all tracks (devices recursive into chains, macros with values, sends), scenes, clip names. Reuses `scripts/als_deep_parse.py`. |
| Rack presets | `adg` | `.adg`, `.adv` | Preset name, rack class, macros (idx + name + value via KeyMidi binding resolution), nested chains, branches. Reuses `scripts/als_deep_parse.py::parse_adg`. |
| Max devices | `amxd` | `.amxd` | Device name + type (audio/instrument/midi), Max version, parser version, exposed Live params (where stored as plain XML, not the frozen JS blob). |
| Plugin presets | `plugin-preset` | `.aupreset`, `.vstpreset`, `.fxp`, `.fxb`, `.nksf` | Plugin name, manufacturer, format (VST/VST3/AU/NKS), preset filename. Param values are plugin-specific binary — opaque, like in `.als` PluginDevice. |

Add your own: register a `Scanner` subclass for any file type you care about.

---

## Manifest — declarative source registry

`~/.livepilot/atlas-overlays/user/manifest.yaml`:

```yaml
schema_version: 1
sources:
  - id: my-projects
    type: als
    path: ~/Music/MyProjects
    recursive: true
    exclude_globs: ["*Backup*", "*.als-archive"]
    last_scanned: 2026-04-30T15:00:00Z
    file_count: 47
    options:
      filename_scale_fallback: true   # use filename key if .als has C-Major default
  - id: my-racks
    type: adg
    path: ~/Music/Ableton/User Library/Presets
    recursive: true
    last_scanned: 2026-04-30T15:30:00Z
    file_count: 1234
  - id: max-devices
    type: amxd
    path: ~/Documents/Max 9/Max for Live Devices
    recursive: true
  - id: vst-presets
    type: plugin-preset
    path: ~/Library/Audio/Presets
    recursive: true
output:
  root: ~/.livepilot/atlas-overlays/user
  schema_version: 1
options:
  parallel_workers: 4
  skip_unchanged: true        # mtime-based incremental scan
  log_level: info
  on_error: continue          # never abort the whole scan over one bad file
ai_annotation:
  enabled: false              # opt-in
  model: sonnet
  fields:
    - sonic_fingerprint
    - tags
    - reach_for
    - avoid
    - cross_references
```

**Why a manifest:** declarative + reproducible + portable. Copy `manifest.yaml` between machines and the same scan reproduces the same corpus. Version-control it if you want corpus-as-code.

---

## Sidecar schema

Every scanner emits two artifacts per source file:

### 1. `_parses/<slug>.json` — full mechanical detail

```json
{
  "schema_version": 1,
  "scanner": "als",
  "scanner_version": "<version>",
  "source_path": "/Users/me/Music/MyProjects/Memphis Tribute.als",
  "source_mtime": 1714477200,
  "source_sha256": "abc123…",
  "scan_timestamp": "2026-04-30T15:00:00Z",
  "data": {
    "bpm": 130.0,
    "scale": {"root_note": "5", "name": "Minor", "source": "filename-fallback"},
    "tracks": [...],
    "scenes": [...]
  }
}
```

The `data` block is whatever the scanner returns — schema is per-type. Mechanical scanners aim for total fidelity to the source file.

### 2. `<slug>.yaml` — searchable thin wrapper

```yaml
entity_id: my-projects__memphis-tribute
entity_type: user_project
namespace: user.projects
name: Memphis Tribute
description: |
  130 BPM, F Minor, 12 tracks, 4 scenes. Rack-heavy: Drift+Saturator,
  Sampler+Erosion, two DrumGroupDevices. Inferred genre: hip-hop.
tags:
- "130bpm"
- "f-minor"
- "minor"
- "12-tracks"
- als-project
- my-projects
- has-drum-rack
- has-sampler
- has-saturator
device_inventory:
- Drift
- Saturator
- Sampler
- DrumGroupDevice
- Erosion
- Reverb
sidecar_path: _parses/memphis-tribute.json
schema_version: 1
```

The wrapper is what `extension_atlas_search` indexes. Tags are auto-derived from the sidecar (`<bpm>bpm`, `<scale-name>`, top device classes, scanner type, source id).

### 3. `annotated/<slug>.yaml` — optional AI annotation

```yaml
entity_id: my-projects__memphis-tribute
namespace: user.projects.annotated
name: Memphis Tribute
sonic_fingerprint: |
  Sample-driven hip-hop sketch. Vinyl-saturated drums with a Memphis-school
  cowbell ride. Bass is a single Drift sub with macro-controlled spectral
  decay. Pad is muted — pulled back to atmosphere only.
reach_for:
- 808-style kick layered with a tight acoustic
- bell as the rhythmic anchor, not the melody
- pad as wash, not a chord
avoid:
- bright synths
- complex chord changes
key_techniques:
- Sampler with chopped vocal phrase, root-mapped to C1
- Saturator on the drum bus driven hard then EQ'd
genre_affinity: ["hip_hop_boom_bap_lo_fi", "memphis_rap"]
producer_anchors: ["DJ Screw", "Mac Mall"]
cross_references:
- my-racks/memphis-bass-rumble.adg  # the bass preset this pulled from
- my-projects/2024-09-beat-sketch.als  # earlier exploration of same idea
schema_version: 1
```

This is AI-written via `corpus_emit_synthesis_briefs`. Same shape as the factory pack identity YAMLs. Optional — the corpus works without it.

---

## Scanner abstraction (write your own)

```python
# mcp_server/user_corpus/scanner.py
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

class Scanner(ABC):
    type_id: ClassVar[str]              # "als", "adg", "amxd", "my-custom"
    file_extensions: ClassVar[list[str]]  # [".als"]
    output_subdir: ClassVar[str]        # "projects"
    schema_version: ClassVar[int] = 1

    @abstractmethod
    def scan_one(self, path: Path) -> dict:
        """Parse one file → JSON-serializable sidecar dict."""

    @abstractmethod
    def derive_tags(self, sidecar: dict) -> list[str]:
        """Extract searchable tags from the sidecar."""

    @abstractmethod
    def derive_description(self, sidecar: dict) -> str:
        """Build a human-readable one-liner."""

    def is_applicable(self, path: Path) -> bool:
        return path.suffix.lower() in self.file_extensions

    def slug(self, path: Path) -> str:
        return path.stem.lower().replace(" ", "-").replace("_", "-")
```

Plug in:

```python
# my_scanner.py
from mcp_server.user_corpus.scanner import Scanner, register_scanner

@register_scanner
class ReaperRppScanner(Scanner):
    type_id = "rpp"
    file_extensions = [".rpp"]
    output_subdir = "reaper_projects"

    def scan_one(self, path):
        # Parse Reaper's .rpp text format
        return {"bpm": ..., "tracks": [...]}

    def derive_tags(self, sidecar):
        return [f"{sidecar['bpm']}bpm", "rpp-project"]

    def derive_description(self, sidecar):
        return f"{sidecar['bpm']} BPM Reaper project, {len(sidecar['tracks'])} tracks"
```

Drop the file in `mcp_server/user_corpus/scanners/` (or anywhere on the Python path), it self-registers via the decorator. The runner picks it up.

---

## Tool workflow (inside Claude Code)

There is no `livepilot corpus` shell command. The corpus is built and queried
through MCP tools you invoke from a Claude Code session — either by naming the
tool directly or by asking in natural language and letting Claude dispatch it.
The corpus tools are: `corpus_setup_wizard`, `corpus_init`, `corpus_add_source`,
`corpus_scan`, `corpus_status`, `corpus_emit_synthesis_briefs`, plus the plugin
helpers (`corpus_detect_plugins`, `corpus_discover_manuals`, …) and
`corpus_list_scanners`.

```
# One-time setup — scaffold the overlay tree + default manifest
> corpus_init
✓ Created ~/.livepilot/atlas-overlays/user/
✓ Wrote default manifest at ~/.livepilot/atlas-overlays/user/manifest.yaml

# (Or run the guided path that wires setup + first sources for you)
> corpus_setup_wizard

# Add sources (writes them into manifest.yaml)
> corpus_add_source(path="~/Music/MyProjects", type_id="als", source_id="my-projects")
✓ Source added: my-projects   (47 files matching .als)

> corpus_add_source(path="~/Music/Ableton/User Library/Presets",
                    type_id="adg", source_id="my-racks", recursive=True)

# Scan everything declared in the manifest
> corpus_scan
[my-projects]   als   47/47
[my-racks]      adg   1234/1234
[max-devices]   amxd  89/89
✓ Wrote 1370 sidecars + wrappers to ~/.livepilot/atlas-overlays/user/

# Scan one source only (incremental — unchanged files are skipped by mtime)
> corpus_scan(source_id="my-racks")

# Status
> corpus_status
manifest: ~/.livepilot/atlas-overlays/user/manifest.yaml (3 sources, 1370 entries)
  my-projects   als   47 files    ✓ fresh
  my-racks      adg   1234 files  ⚠ 12 files newer than scan
  max-devices   amxd  89 files    ✓ fresh

# AI annotation — emit synthesis briefs for sonnet subagents to fill in
> corpus_emit_synthesis_briefs(source_id="my-projects")

# Search — use the atlas search surface against the user namespace
> extension_atlas_search(query="warm pad", namespace="user.racks")
3 matches:
  0.78  Frozen Spectrum.adg
  0.71  Glacier Pad.adg
  0.68  Polar Drift.adg
```

---

## Inside Claude Code

Once the corpus is built, every existing atlas tool works on it:

```
> Find me a warm bass preset like the one in 2024-09-beat-sketch
[Claude calls atlas_macro_fingerprint with source_pack_slug='user.my-projects', ...]
[returns 5 matches, top hit is in user.my-racks/analog-bass-rumble.adg]

> What's the harmonic spine of Memphis Tribute?
[Claude calls extension_atlas_get(namespace='user.projects', entity_id='memphis-tribute')]
[reads the JSON sidecar + annotated YAML, summarizes]

> Compose me a beat in the style of my recent sketches
[Claude calls atlas_pack_aware_compose with brief built from
 cross-references in annotated/*.yaml across user.projects]
```

The user's corpus is a first-class citizen alongside the factory atlas.

---

## AI annotation flow

1. User runs `corpus_emit_synthesis_briefs(source_id="my-projects")` from Claude Code.
2. The MCP tool reads all `_parses/*.json` for that source.
3. For each sidecar, it emits a synthesis brief bundling:
   - The sidecar contents (BPM, devices, structure)
   - The producer-vocabulary references from `livepilot/skills/livepilot-core/references/`
   - A YAML schema to fill in
4. A sonnet subagent works each brief and writes one `annotated/<slug>.yaml`.
5. Tool reports counts + any failures.
6. User can re-run on subsets, override fields manually, or skip annotation entirely.

This pattern matches what we used for the 44 factory pack identity YAMLs — sonnet handles prose-shaped output that regex can't extract.

---

## Privacy + portability

- `~/.livepilot/atlas-overlays/user/` is gitignored at the LivePilot repo root. Your corpus stays local unless you choose to share it.
- The corpus is plain JSON + YAML. Easy to inspect, diff, edit by hand, or version-control privately.
- No telemetry. The scanner reads files locally; nothing is sent anywhere.
- Sidecars include a `source_path` field — strip these if you ever publish a corpus.
- `manifest.yaml` is the only file you'd ever want to share — it has paths but not content.

---

## Performance characteristics

For reference, on this machine:
- 104 `.als` files: 13s
- 3,813 `.adg` files: 52s
- Single .als (10MB compressed): 100-150ms
- Single .adg (50KB): 14ms median

A typical user corpus of:
- 50 `.als` projects (~1.5GB) → ~7s
- 1,200 `.adg` racks → ~17s
- 100 `.amxd` devices → ~3s
- 5,000 plugin presets → ~12s

≈ **40 seconds for a one-time full scan**, then incremental rescans skip unchanged files.

---

## What gets indexed for search

The wrapper YAML is what `extension_atlas_search` ranks. Default tag derivation:

| Source | Auto-tags |
|--------|-----------|
| .als | `<bpm>bpm`, scale name, scale root, track count, top 5 device classes, scanner id, source id |
| .adg | rack class (instrument/audio_effect/...), top 3 macro names (after Unicode-fold), nested-rack flag |
| .amxd | device type (audio/instrument/midi), Max version, top exposed param names |
| plugin-preset | manufacturer, format (vst/vst3/au), file extension |

Override or extend in your scanner's `derive_tags(sidecar)` method.

---

## Migration / upgrade path

`schema_version` lives in every sidecar + the manifest. Future scanner upgrades:

1. Bump scanner's `schema_version` constant.
2. Old sidecars still load; the runner detects mismatch on incremental scan and rescans those files.
3. Migration scripts under `mcp_server/user_corpus/` can transform old → new schema if needed.

The factory atlas already lives by this rule (its schema bumped 4 times during v1.23.4 fix waves) — same discipline applies.

---

## Where to start

All steps run from a Claude Code session — by tool call or in natural language:

1. `corpus_init` (or `corpus_setup_wizard` for the guided path)
2. `corpus_add_source(path="~/Music/Ableton/User Library", type_id="adg", source_id="my-racks", recursive=True)`
3. `corpus_scan`
4. `extension_atlas_search(query="warm pad", namespace="user.racks")` — verify the corpus is queryable
5. (Optional) `corpus_emit_synthesis_briefs(source_id="my-racks")` — turn mechanical sidecars into described knowledge
6. Iterate: add more sources, tune annotations, write your own scanner for unusual content

You're done when you can ask Claude *"what's a good bass to layer with the kick from 2024-12-foo.als"* and get a real, corpus-grounded answer.

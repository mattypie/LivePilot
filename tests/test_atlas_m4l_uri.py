"""LIVE#3 — M4L pack-instrument URI guard in atlas_search / atlas_suggest.

The bug: for M4L pack instruments like Tree Tone (Inspired by Nature),
atlas_search returns uri='query:Synths#Tree%20Tone'.  That URI scheme only
resolves for *native* Ableton synths (Operator, Wavetable, etc.).
load_browser_item with the Synths# URI fails with INVALID_PARAM in Live
because M4L pack instruments are not browsable under "Synths" — only their
.adg presets are browsable under "sounds".

The fix (_patch_m4l_uri in tools.py):
  - Clears the bogus query:Synths# URI (sets it to "")
  - Sets load_via = "preset"
  - Adds browse_hint directing the agent to search_browser(path="sounds")

Scope: _patch_m4l_uri + the factory-result loop in atlas_search.
       Native synths (Operator, Wavetable, Drift …) must NOT be affected.
"""

from __future__ import annotations

import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from mcp_server.atlas import tools as atlas_tools

# ── _patch_m4l_uri direct unit tests ─────────────────────────────────────────


class TestPatchM4lUri:
    """Unit tests for the _patch_m4l_uri helper function."""

    # ── M4L pack instruments — URI must be cleared ────────────────────

    @pytest.mark.parametrize("dev_id,name", [
        ("tree_tone",    "Tree Tone"),
        ("vector_fm",    "Vector FM"),
        ("vector_grain", "Vector Grain"),
        ("emit",         "Emit"),
    ])
    def test_m4l_pack_synth_uri_cleared(self, dev_id: str, name: str):
        """Known M4L pack instruments must NOT advertise a query:Synths# URI."""
        device = {
            "id": dev_id,
            "name": name,
            "uri": f"query:Synths#{name.replace(' ', '%20')}",
            "category": "instruments",
        }
        entry = {
            "id": dev_id,
            "name": name,
            "uri": f"query:Synths#{name.replace(' ', '%20')}",
        }
        result = atlas_tools._patch_m4l_uri(entry, device)

        # URI must be cleared — not the broken Synths# value
        assert result["uri"] == "", (
            f"{name}: uri should be cleared but got {result['uri']!r}"
        )
        assert not result["uri"].startswith("query:Synths#"), (
            f"{name}: must NOT expose a query:Synths# URI"
        )

    @pytest.mark.parametrize("dev_id,name", [
        ("tree_tone",    "Tree Tone"),
        ("vector_fm",    "Vector FM"),
        ("vector_grain", "Vector Grain"),
        ("emit",         "Emit"),
    ])
    def test_m4l_pack_synth_gets_load_via_preset(self, dev_id: str, name: str):
        """load_via='preset' must be set so agents know to go via search_browser."""
        device = {
            "id": dev_id,
            "name": name,
            "uri": f"query:Synths#{name.replace(' ', '%20')}",
        }
        entry = dict(uri=f"query:Synths#{name.replace(' ', '%20')}")
        result = atlas_tools._patch_m4l_uri(entry, device)

        assert result.get("load_via") == "preset"

    @pytest.mark.parametrize("dev_id,name", [
        ("tree_tone",    "Tree Tone"),
        ("vector_fm",    "Vector FM"),
        ("vector_grain", "Vector Grain"),
        ("emit",         "Emit"),
    ])
    def test_m4l_pack_synth_browse_hint_sounds_path(self, dev_id: str, name: str):
        """browse_hint must point agents at the 'sounds' browser path."""
        device = {
            "id": dev_id,
            "name": name,
            "uri": f"query:Synths#{name.replace(' ', '%20')}",
        }
        entry = dict(uri=f"query:Synths#{name.replace(' ', '%20')}")
        result = atlas_tools._patch_m4l_uri(entry, device)

        hint = result.get("browse_hint")
        assert hint is not None, f"{name}: browse_hint must be present"
        assert hint.get("path") == "sounds", (
            f"{name}: browse_hint.path must be 'sounds', got {hint.get('path')!r}"
        )
        assert hint.get("name_filter") == name, (
            f"{name}: browse_hint.name_filter must be the device name"
        )
        assert "note" in hint, f"{name}: browse_hint must include a note"

    # ── Native synths — URI must be untouched ─────────────────────────

    @pytest.mark.parametrize("dev_id,name,uri", [
        ("operator",   "Operator",   "query:Synths#Operator"),
        ("wavetable",  "Wavetable",  "query:Synths#Wavetable"),
        ("drift",      "Drift",      "query:Synths#Drift"),
        ("collision",  "Collision",  "query:Synths#Collision"),
        ("analog",     "Analog",     "query:Synths#Analog"),
        ("meld",       "Meld",       "query:Synths#Meld"),
        ("poli",       "Poli",       "query:Synths#Poli"),
        ("sampler",    "Sampler",    "query:Synths#Sampler"),
        ("simpler",    "Simpler",    "query:Synths#Simpler"),
        ("impulse",    "Impulse",    "query:Synths#Impulse"),
        ("granulator_iii", "Granulator III", "query:Synths#Granulator%20III"),
    ])
    def test_native_synth_uri_unchanged(self, dev_id: str, name: str, uri: str):
        """Native instruments must keep their query:Synths# URI unchanged."""
        device = {"id": dev_id, "name": name, "uri": uri}
        entry = dict(uri=uri)
        result = atlas_tools._patch_m4l_uri(entry, device)

        assert result["uri"] == uri, (
            f"{name}: native synth URI should be unchanged, got {result['uri']!r}"
        )
        assert "load_via" not in result, (
            f"{name}: load_via must NOT be added to native synths"
        )
        assert "browse_hint" not in result, (
            f"{name}: browse_hint must NOT be added to native synths"
        )

    def test_non_synths_uri_untouched(self):
        """A query:Sounds# or other URI on an M4L ID must not be altered."""
        device = {"id": "tree_tone", "name": "Tree Tone", "uri": "query:Sounds#Tree%20Tone"}
        entry = dict(uri="query:Sounds#Tree%20Tone")
        result = atlas_tools._patch_m4l_uri(entry, device)
        # The guard is on query:Synths# — a Sounds# URI should not be touched
        assert result["uri"] == "query:Sounds#Tree%20Tone"
        assert "load_via" not in result

    def test_empty_id_not_patched(self):
        """A device with no ID and a Synths# URI must not be patched."""
        device = {"id": "", "name": "Unknown", "uri": "query:Synths#Unknown"}
        entry = dict(uri="query:Synths#Unknown")
        result = atlas_tools._patch_m4l_uri(entry, device)
        assert result["uri"] == "query:Synths#Unknown"

    def test_returns_same_dict_reference(self):
        """_patch_m4l_uri mutates and returns the same dict object."""
        device = {"id": "tree_tone", "name": "Tree Tone",
                  "uri": "query:Synths#Tree%20Tone"}
        entry: dict = {"uri": "query:Synths#Tree%20Tone"}
        returned = atlas_tools._patch_m4l_uri(entry, device)
        assert returned is entry


# ── Integration: _patch_m4l_uri called through atlas_search ──────────────────

def _make_minimal_atlas_json(devices: list[dict]) -> str:
    """Return a JSON string for a minimal atlas fixture."""
    return json.dumps({"version": "2.0.0", "devices": devices})


def _make_atlas_file(tmp_path, devices: list[dict]) -> str:
    path = tmp_path / "device_atlas.json"
    path.write_text(_make_minimal_atlas_json(devices))
    return str(path)


TREE_TONE_DEVICE = {
    "id": "tree_tone",
    "name": "Tree Tone",
    "uri": "query:Synths#Tree%20Tone",
    "category": "instruments",
    "subcategory": "synths",
    "source": "native",
    "enriched": True,
    "character_tags": ["organic", "evolving"],
    "use_cases": ["pads", "textures"],
    "genre_affinity": {"primary": ["ambient"], "secondary": []},
    "self_contained": True,
    "sonic_description": "Multi-layer tonal synthesis instrument.",
    "complexity": "intermediate",
    "synthesis_type": "additive",
}

OPERATOR_DEVICE = {
    "id": "operator",
    "name": "Operator",
    "uri": "query:Synths#Operator",
    "category": "instruments",
    "subcategory": "synths",
    "source": "native",
    "enriched": True,
    "character_tags": ["fm", "digital", "complex"],
    "use_cases": ["bass", "leads", "pads"],
    "genre_affinity": {"primary": ["electronic"], "secondary": []},
    "self_contained": True,
    "sonic_description": "Four-operator FM synthesizer.",
    "complexity": "advanced",
    "synthesis_type": "fm",
}


@pytest.fixture
def atlas_with_tree_tone_and_operator(tmp_path):
    """Returns a loaded AtlasManager instance with Tree Tone + Operator."""
    from mcp_server.atlas import AtlasManager
    path = _make_atlas_file(tmp_path, [TREE_TONE_DEVICE, OPERATOR_DEVICE])
    return AtlasManager(path)


def _run_atlas_search(atlas_manager, query: str, category: str = "all") -> list[dict]:
    """Call the factory-result path of atlas_search using the given AtlasManager.

    Bypasses the FastMCP context by calling the internal atlas.search() then
    running results through the same _patch_m4l_uri loop that atlas_search does.
    This tests the integration without standing up the full MCP server.
    """
    raw_results = atlas_manager.search(query, category=category, limit=10)
    patched = []
    for r in raw_results:
        dev = r["device"]
        enriched = bool(dev.get("enriched", False))
        entry: dict = {
            "id": dev.get("id", ""),
            "name": dev.get("name", ""),
            "uri": dev.get("uri", ""),
            "category": dev.get("category", ""),
            "sonic_description": dev.get("sonic_description", "")[:400],
            "character_tags": dev.get("character_tags", [])[:5],
            "enriched": enriched,
            "score": r.get("score", 0),
            "source": "factory_atlas",
        }
        if enriched:
            entry.update(atlas_tools._surface_enriched_fields(dev))
        atlas_tools._patch_m4l_uri(entry, dev)
        patched.append(entry)
    return patched


class TestAtlasSearchM4lUriIntegration:
    """Integration tests: Tree Tone search returns no query:Synths# URI."""

    def test_tree_tone_search_has_no_synths_uri(self, atlas_with_tree_tone_and_operator):
        results = _run_atlas_search(atlas_with_tree_tone_and_operator, "tree tone")
        tree_results = [r for r in results if r["id"] == "tree_tone"]
        assert tree_results, "Tree Tone must appear in search results"
        r = tree_results[0]
        assert r["uri"] == "", (
            f"Tree Tone must not expose query:Synths# URI, got {r['uri']!r}"
        )
        assert not r["uri"].startswith("query:Synths#")

    def test_tree_tone_search_has_load_via_preset(self, atlas_with_tree_tone_and_operator):
        results = _run_atlas_search(atlas_with_tree_tone_and_operator, "tree tone")
        tree_results = [r for r in results if r["id"] == "tree_tone"]
        assert tree_results
        assert tree_results[0].get("load_via") == "preset"

    def test_tree_tone_search_has_browse_hint(self, atlas_with_tree_tone_and_operator):
        results = _run_atlas_search(atlas_with_tree_tone_and_operator, "tree tone")
        tree_results = [r for r in results if r["id"] == "tree_tone"]
        assert tree_results
        hint = tree_results[0].get("browse_hint")
        assert hint is not None
        assert hint["path"] == "sounds"
        assert "Tree Tone" in hint["name_filter"]

    def test_operator_search_keeps_synths_uri(self, atlas_with_tree_tone_and_operator):
        """Operator (native synth) must keep its query:Synths#Operator URI."""
        results = _run_atlas_search(atlas_with_tree_tone_and_operator, "operator fm")
        op_results = [r for r in results if r["id"] == "operator"]
        assert op_results, "Operator must appear in search results"
        r = op_results[0]
        assert r["uri"] == "query:Synths#Operator", (
            f"Operator must keep its native URI, got {r['uri']!r}"
        )
        assert "load_via" not in r
        assert "browse_hint" not in r

    def test_both_devices_same_search_independent(self, atlas_with_tree_tone_and_operator):
        """Broad search returning both devices: Tree Tone patched, Operator intact."""
        results = _run_atlas_search(atlas_with_tree_tone_and_operator, "synth")
        by_id = {r["id"]: r for r in results}

        if "tree_tone" in by_id:
            assert by_id["tree_tone"]["uri"] == ""
            assert by_id["tree_tone"].get("load_via") == "preset"

        if "operator" in by_id:
            assert by_id["operator"]["uri"] == "query:Synths#Operator"
            assert "load_via" not in by_id["operator"]


# ── M4L_PACK_SYNTH_IDS constant completeness check ───────────────────────────

class TestM4lPackSynthIdsConstant:
    """Verify the constant itself is correctly populated."""

    def test_all_four_inspired_by_nature_instruments_in_set(self):
        ids = atlas_tools._M4L_PACK_SYNTH_IDS
        for expected in ("tree_tone", "vector_fm", "vector_grain", "emit"):
            assert expected in ids, f"{expected!r} must be in _M4L_PACK_SYNTH_IDS"

    def test_native_synths_not_in_set(self):
        ids = atlas_tools._M4L_PACK_SYNTH_IDS
        for native in ("operator", "wavetable", "drift", "collision", "analog",
                       "meld", "poli", "simpler", "sampler", "impulse",
                       "granulator_iii"):
            assert native not in ids, (
                f"{native!r} must NOT be in _M4L_PACK_SYNTH_IDS — it is a native synth"
            )

    def test_set_is_frozenset(self):
        assert isinstance(atlas_tools._M4L_PACK_SYNTH_IDS, frozenset)


# ── Integration: _patch_m4l_uri called through atlas_device_info (LIVE#3) ─────


class TestAtlasDeviceInfoM4lUri:
    """LIVE#3 completeness — a DIRECT atlas_device_info lookup (not just
    atlas_search) must also clear the bogus query:Synths# URI for M4L pack
    instruments, and must NOT mutate the live in-memory atlas record."""

    def test_device_info_verbose_clears_m4l_uri(self, atlas_with_tree_tone_and_operator):
        mgr = atlas_with_tree_tone_and_operator
        with patch.object(atlas_tools, "_get_atlas", return_value=mgr):
            result = atlas_tools.atlas_device_info(ctx=None, device_id="tree_tone", verbose=True)
        assert result["uri"] == ""
        assert result["load_via"] == "preset"
        assert result["browse_hint"]["path"] == "sounds"
        # The live atlas record must be untouched (we patched a copy).
        assert mgr.lookup("tree_tone")["uri"] == "query:Synths#Tree%20Tone"

    def test_device_info_compact_clears_m4l_uri(self, atlas_with_tree_tone_and_operator):
        mgr = atlas_with_tree_tone_and_operator
        with patch.object(atlas_tools, "_get_atlas", return_value=mgr):
            result = atlas_tools.atlas_device_info(ctx=None, device_id="tree_tone", verbose=False)
        assert result["uri"] == ""
        assert result["load_via"] == "preset"
        assert mgr.lookup("tree_tone")["uri"] == "query:Synths#Tree%20Tone"

    def test_device_info_native_synth_uri_unchanged(self, atlas_with_tree_tone_and_operator):
        mgr = atlas_with_tree_tone_and_operator
        with patch.object(atlas_tools, "_get_atlas", return_value=mgr):
            result = atlas_tools.atlas_device_info(ctx=None, device_id="operator", verbose=True)
        assert result["uri"] == "query:Synths#Operator"
        assert "load_via" not in result


# ── Integration: REAL atlas_search / atlas_suggest / atlas_describe_chain ─────
# (LIVE3-3) These call the actual tool functions so a removed _patch_m4l_uri
# call at the real call sites would fail the suite — closing the mutation blind
# spot where the prior tests reimplemented the loop.


class TestRealAtlasToolM4lUri:
    def test_real_atlas_search_clears_m4l_uri(self, atlas_with_tree_tone_and_operator):
        mgr = atlas_with_tree_tone_and_operator
        with patch.object(atlas_tools, "_get_atlas", return_value=mgr):
            result = atlas_tools.atlas_search(ctx=None, query="tree tone")
        results = result.get("results", [])
        tree = next((r for r in results if r.get("id") == "tree_tone"
                     or r.get("name") == "Tree Tone"), None)
        assert tree is not None, f"tree_tone not in results: {[r.get('id') for r in results]}"
        assert tree["uri"] == ""
        assert tree.get("load_via") == "preset"

    def test_real_atlas_search_keeps_native_uri(self, atlas_with_tree_tone_and_operator):
        mgr = atlas_with_tree_tone_and_operator
        with patch.object(atlas_tools, "_get_atlas", return_value=mgr):
            result = atlas_tools.atlas_search(ctx=None, query="operator fm")
        results = result.get("results", [])
        op = next((r for r in results if r.get("id") == "operator"), None)
        if op is not None:
            assert op["uri"] == "query:Synths#Operator"
            assert "load_via" not in op

    def test_real_atlas_suggest_clears_m4l_uri(self, atlas_with_tree_tone_and_operator):
        mgr = atlas_with_tree_tone_and_operator
        with patch.object(atlas_tools, "_get_atlas", return_value=mgr):
            result = atlas_tools.atlas_suggest(ctx=None, intent="additive pad organic")
        # Find tree_tone anywhere in the suggestion payload.
        import json as _json
        blob = _json.dumps(result)
        if "tree_tone" in blob or "Tree Tone" in blob:
            assert "query:Synths#Tree" not in blob, (
                "atlas_suggest leaked a bogus query:Synths# URI for an M4L pack instrument")

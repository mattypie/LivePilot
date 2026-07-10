"""Sidechain routing-source matching ladder (remote_script/LivePilot/mixing.py).

Pins the three-tier match in _match_routing_type, including the
case-insensitive EXACT tier added after a live Ableton 12.4.2 session showed
routing menus can list a bare, unprefixed track name ("Kick") — where the
"-<name>" suffix fallback never fires and a lowercase caller previously got
INVALID_PARAM (found live 2026-07-10).
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parent.parent
MIXING = REPO / "remote_script" / "LivePilot" / "mixing.py"


def _load_mixing():
    """Load mixing.py in isolation with the router's @register stubbed."""
    router = types.ModuleType("LivePilot.router")

    def register(name):
        def deco(fn):
            return fn
        return deco

    router.register = register
    pkg = types.ModuleType("LivePilot")
    pkg.__path__ = [str(MIXING.parent)]
    sys.modules.setdefault("LivePilot", pkg)
    sys.modules["LivePilot.router"] = router

    spec = importlib.util.spec_from_file_location("LivePilot.mixing", MIXING)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mixing = _load_mixing()


def _menu(*names):
    return [SimpleNamespace(display_name=n) for n in names]


def test_exact_match_wins():
    menu = _menu("1-Kick", "Kick", "Main")
    assert _mixing._match_routing_type(menu, "1-Kick").display_name == "1-Kick"


def test_case_insensitive_exact_match_unprefixed_menu():
    # The live-found case: Live 12.4.2 shows a bare "Kick" entry; lowercase
    # caller must resolve (suffix rule can't fire without a "-" prefix).
    menu = _menu("Kick", "4-Audio", "Main", "No Input")
    assert _mixing._match_routing_type(menu, "kick").display_name == "Kick"


def test_suffix_match_prefixed_menu_casefolded():
    menu = _menu("1-KICK", "2-Bass", "Main")
    assert _mixing._match_routing_type(menu, "kick").display_name == "1-KICK"


def test_exact_beats_suffix_when_both_present():
    # An exact "Kick" entry must win over a "3-Kick" suffix candidate.
    menu = _menu("3-Kick", "Kick")
    assert _mixing._match_routing_type(menu, "Kick").display_name == "Kick"


def test_no_match_returns_none():
    menu = _menu("1-Bass", "Main", "No Input")
    assert _mixing._match_routing_type(menu, "kick") is None

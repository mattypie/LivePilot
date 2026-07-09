"""Regression test for P2-48 — undo/redo must stay write-classified even if
WRITE_COMMANDS is ever trimmed.

"undo"/"redo" match no read/write prefix (READ_COMMAND_PREFIXES or
WRITE_COMMAND_PREFIXES), so before this fix the ONLY thing keeping them
write-classified was their literal presence in WRITE_COMMANDS. A future
well-intentioned dedupe/trim of that set could silently reclassify them as
reads, reintroducing the read-after-write race the write timeout + settle
delay exist to prevent. WRITE_BARE_VERBS is an independent backstop that
must catch them regardless of what WRITE_COMMANDS contains.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = ROOT / "remote_script" / "LivePilot"


def _load_server_module():
    for name in [
        "remote_script.LivePilot.server",
        "remote_script.LivePilot.router",
        "remote_script.LivePilot.utils",
        "remote_script.LivePilot",
        "remote_script",
    ]:
        sys.modules.pop(name, None)

    remote_pkg = types.ModuleType("remote_script")
    remote_pkg.__path__ = [str(ROOT / "remote_script")]
    sys.modules["remote_script"] = remote_pkg

    live_pkg = types.ModuleType("remote_script.LivePilot")
    live_pkg.__path__ = [str(REMOTE_ROOT)]
    sys.modules["remote_script.LivePilot"] = live_pkg

    def _load(name: str, path: Path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module

    _load("remote_script.LivePilot.utils", REMOTE_ROOT / "utils.py")
    _load("remote_script.LivePilot.router", REMOTE_ROOT / "router.py")
    return _load("remote_script.LivePilot.server", REMOTE_ROOT / "server.py")


def test_write_bare_verbs_frozenset_contains_undo_redo():
    server_mod = _load_server_module()
    assert "undo" in server_mod.WRITE_BARE_VERBS
    assert "redo" in server_mod.WRITE_BARE_VERBS


def test_undo_redo_classify_as_write_even_if_removed_from_write_commands():
    """Simulate a future trim of WRITE_COMMANDS that drops undo/redo.

    is_write_command() must still classify them as writes because
    WRITE_BARE_VERBS is checked independently — this is the actual
    regression guard for P2-48, not just a presence check on the set.
    """
    server_mod = _load_server_module()

    trimmed = server_mod.WRITE_COMMANDS - {"undo", "redo"}
    assert "undo" not in trimmed and "redo" not in trimmed
    server_mod.WRITE_COMMANDS = trimmed

    assert server_mod.is_write_command("undo") is True
    assert server_mod.is_write_command("redo") is True


def test_write_bare_verbs_backstop_does_not_affect_unrelated_commands():
    """The backstop must not turn unrelated bare-verb-shaped commands into
    false-positive writes — it only ever contains undo/redo."""
    server_mod = _load_server_module()
    assert server_mod.is_write_command("ping") is False
    assert server_mod.is_write_command("get_session_info") is False

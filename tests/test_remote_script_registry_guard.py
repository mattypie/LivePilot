"""Guard-rail: no duplicate @register("name") command types in remote_script.

router.py's registry (remote_script/LivePilot/router.py) is a plain dict:

    def register(command_type):
        def decorator(fn):
            _handlers[command_type] = fn
            return fn
        return decorator

A second ``@register("same_name")`` anywhere in remote_script/LivePilot/*.py
silently overwrites the first handler at import time — no exception, no
warning, just a shadowed handler that the MCP server can never reach again.
That's a much nastier failure mode than a normal Python NameError because
dispatch still succeeds; it just quietly runs the wrong code.

This scans the AST of every remote_script/LivePilot/*.py module (no import,
no live Ableton/Live dependency needed) and collects every string literal
passed to a ``@register(...)`` decorator. Currently zero collisions — this
test exists purely to catch the day someone copy-pastes a handler and
forgets to rename the command_type.

Mirrors the Counter-based pattern in test_registry_uniqueness.py (which
guards the semantic_moves registry the same way).
"""

from __future__ import annotations

import ast
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = ROOT / "remote_script" / "LivePilot"


def _collect_register_calls() -> dict[str, list[str]]:
    """Return {command_type: [file:lineno, ...]} for every @register(...) call."""
    locations: dict[str, list[str]] = defaultdict(list)

    for path in sorted(REMOTE_ROOT.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for deco in node.decorator_list:
                if not isinstance(deco, ast.Call):
                    continue
                func = deco.func
                func_name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
                if func_name != "register":
                    continue
                if not deco.args or not isinstance(deco.args[0], ast.Constant):
                    continue
                command_type = deco.args[0].value
                if not isinstance(command_type, str):
                    continue
                locations[command_type].append(f"{path.name}:{node.lineno}")

    return locations


def test_remote_script_directory_has_python_files():
    """Sanity check the glob actually found the handler modules — an empty
    scan would make every assertion below vacuously true."""
    assert list(REMOTE_ROOT.glob("*.py")), f"No .py files found under {REMOTE_ROOT}"


def test_no_duplicate_register_command_types():
    locations = _collect_register_calls()
    counts = Counter({name: len(locs) for name, locs in locations.items()})
    dupes = {name: locations[name] for name, n in counts.items() if n > 1}
    assert not dupes, (
        f"Duplicate @register(...) command types would silently shadow the "
        f"earlier handler (router._handlers is a plain dict): {dupes}"
    )


def test_every_register_call_has_a_nonempty_string_name():
    locations = _collect_register_calls()
    assert locations, "Expected at least one @register(...) handler to be found"
    for name in locations:
        assert isinstance(name, str) and name, f"Empty/invalid command_type: {name!r}"


def test_registered_command_count_meets_baseline():
    """Loose floor so an accidental deletion of a whole handler file's
    decorators (e.g. a bad merge) doesn't slip through silently. Not tied
    to the exact tool-count constants elsewhere — just a sanity floor."""
    locations = _collect_register_calls()
    assert len(locations) >= 150, (
        f"Registered remote_script command count dropped to {len(locations)}; "
        f"expected at least 150. Check for an accidental handler-file "
        f"truncation or a broken merge."
    )

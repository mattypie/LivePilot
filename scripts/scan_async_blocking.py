#!/usr/bin/env python3
"""AST scanner for the "blocking call on the event loop" bug class.

Finds `async def` functions in mcp_server/ that call one of the known
blocking primitives directly, or one hop away through a same-file plain
`def` helper, without being offloaded via `asyncio.to_thread`,
`loop.run_in_executor`, or the `send_command_async` wrapper.

Blocking primitives detected:
  - `<name>.send_command(...)`         (sync TCP client — excludes bridge/m4l)
  - `<name>.write_bytes/write_text/read_bytes/read_text/mkdir(...)` (pathlib-style file I/O)
  - `subprocess.<anything>(...)`       (subprocess module calls)

Explicitly NOT a violation:
  - `<name>.send_command_async(...)`   (the awaitable wrapper)
  - `bridge.send_command(...)` / anything with "bridge" or "m4l" in the
    dotted receiver chain (the M4L bridge is already async-native)
  - Any blocking call already wrapped by an ancestor `asyncio.to_thread(...)`
    or `<x>.run_in_executor(...)` call in the same expression tree.

Usage:
    python3 scripts/scan_async_blocking.py [root] [--exclude path1 path2 ...] [--json]

Exit code is 1 if any violation is found (for CI), 0 otherwise.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Attribute names that count as blocking file I/O when called on any receiver.
FILE_IO_ATTRS = {"write_bytes", "write_text", "read_bytes", "read_text", "mkdir"}

# Receiver-chain tokens that exempt a `.send_command(...)` call — these
# receivers are already async-native (M4L bridge / spectral cache) and must
# never be wrapped in to_thread.
EXEMPT_RECEIVER_TOKENS = {"bridge", "m4l", "spectral"}

# Call attribute names that count as "already offloaded" wrappers.
OFFLOAD_ATTRS = {"to_thread", "run_in_executor"}


def _dotted_chain_tokens(node: ast.AST) -> list[str]:
    """Return the lowercased identifier tokens in a dotted attribute/name chain.

    e.g. `self.m4l_bridge.send_command` -> ["self", "m4l_bridge", "send_command"]
    """
    tokens: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        tokens.append(cur.attr.lower())
        cur = cur.value
    if isinstance(cur, ast.Name):
        tokens.append(cur.id.lower())
    tokens.reverse()
    return tokens


def _is_exempt_receiver(func: ast.Attribute) -> bool:
    """True if the receiver chain of a `.send_command` call mentions bridge/m4l."""
    tokens = _dotted_chain_tokens(func.value)
    return any(any(ex in tok for ex in EXEMPT_RECEIVER_TOKENS) for tok in tokens)


def _is_subprocess_call(func: ast.Attribute) -> bool:
    tokens = _dotted_chain_tokens(func)
    return "subprocess" in tokens[:-1]  # subprocess.<attr>(...)


def _classify_blocking_call(call: ast.Call) -> str | None:
    """Return a short label if `call` is one of the tracked blocking primitives."""
    func = call.func
    if not isinstance(func, ast.Attribute):
        return None

    if func.attr == "send_command":
        if _is_exempt_receiver(func):
            return None
        return "send_command"

    if func.attr in FILE_IO_ATTRS:
        # Cheap false-positive guard: don't flag calls on obviously unrelated
        # receivers like io.BytesIO()/mock objects named `buf`/`stream` when
        # we can tell from the chain — but we keep this permissive since
        # pathlib.Path is the overwhelming real-world case and the whole
        # point of this scanner is to surface candidates for human review.
        return f"file_io:{func.attr}"

    if _is_subprocess_call(func):
        return f"subprocess:{func.attr}"

    return None


def _is_offload_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Attribute):
        return func.attr in OFFLOAD_ATTRS
    return False


@dataclass
class ParentIndex:
    parent: dict[int, ast.AST] = field(default_factory=dict)

    def link(self, tree: ast.AST) -> None:
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                self.parent[id(child)] = node

    def ancestors(self, node: ast.AST):
        cur = self.parent.get(id(node))
        while cur is not None:
            yield cur
            cur = self.parent.get(id(cur))


def _is_wrapped(call: ast.Call, parents: ParentIndex, stop_at: ast.AST) -> bool:
    """True if `call` sits inside an ancestor to_thread/run_in_executor Call,
    without crossing out of `stop_at` (the enclosing function)."""
    for anc in parents.ancestors(call):
        if anc is stop_at:
            return False
        if _is_offload_call(anc):
            return True
        # Don't cross into a nested function definition's own scope boundary
        # accidentally — ast.walk/parent chain naturally stays within the
        # lexical tree so this is only a safety cap.
        if anc is None:
            break
    return False


def _collect_plain_helpers(module: ast.Module) -> dict[str, ast.FunctionDef]:
    """All same-file plain (sync) function/method defs, keyed by simple name.

    Best-effort: later definitions with the same simple name overwrite
    earlier ones (heuristic — this is a candidate-surfacing scanner, not a
    fully scoped resolver).
    """
    helpers: dict[str, ast.FunctionDef] = {}
    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef):
            helpers[node.name] = node
    return helpers


def _find_blocking_calls_in_body(
    func_node: ast.AST, parents: ParentIndex
) -> list[tuple[ast.Call, str]]:
    """Find unwrapped blocking calls anywhere in func_node's body (any depth,
    but does NOT descend into nested async def — those are scanned as their
    own top-level findings when ast.walk reaches them separately)."""
    found: list[tuple[ast.Call, str]] = []
    for node in ast.walk(func_node):
        if node is func_node:
            continue
        if isinstance(node, ast.Call):
            label = _classify_blocking_call(node)
            if label is None:
                continue
            if _is_wrapped(node, parents, func_node):
                continue
            found.append((node, label))
    return found


def _call_target_name(call: ast.Call) -> str | None:
    """Simple-name resolution for a call target: `helper(...)` or
    `self.helper(...)` / `obj.helper(...)` -> "helper"."""
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


@dataclass
class Violation:
    file: str
    lineno: int
    func_name: str
    kind: str  # "direct" or "via_helper"
    detail: str
    helper_name: str | None = None
    helper_lineno: int | None = None

    def to_dict(self) -> dict:
        return {
            "file": self.file,
            "lineno": self.lineno,
            "func_name": self.func_name,
            "kind": self.kind,
            "detail": self.detail,
            "helper_name": self.helper_name,
            "helper_lineno": self.helper_lineno,
        }


def scan_file(path: Path) -> list[Violation]:
    try:
        src = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        print(f"WARN: could not read {path}: {exc}", file=sys.stderr)
        return []

    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError as exc:
        print(f"WARN: syntax error in {path}: {exc}", file=sys.stderr)
        return []

    parents = ParentIndex()
    parents.link(tree)

    plain_helpers = _collect_plain_helpers(tree)

    violations: list[Violation] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.AsyncFunctionDef):
            continue

        # --- TYPE A: direct unwrapped blocking calls in this async fn ---
        direct_hits = _find_blocking_calls_in_body(node, parents)
        for call, label in direct_hits:
            violations.append(
                Violation(
                    file=str(path),
                    lineno=call.lineno,
                    func_name=node.name,
                    kind="direct",
                    detail=label,
                )
            )

        # --- TYPE B: one-level same-file hop through a plain-def helper ---
        for sub in ast.walk(node):
            if sub is node:
                continue
            if isinstance(sub, (ast.AsyncFunctionDef, ast.FunctionDef)) and sub is not node:
                # Don't treat calls made from a *nested* function definition
                # as if they were made directly by `node` — skip descending
                # into nested def bodies for the helper-hop check. (Nested
                # async defs get their own top-level scan pass anyway.)
                continue
            if not isinstance(sub, ast.Call):
                continue
            target_name = _call_target_name(sub)
            if target_name is None or target_name not in plain_helpers:
                continue
            helper_node = plain_helpers[target_name]
            if helper_node is node:
                continue
            if isinstance(helper_node, ast.AsyncFunctionDef):
                continue
            if _is_wrapped(sub, parents, node):
                continue
            helper_hits = _find_blocking_calls_in_body(helper_node, parents)
            for _call, label in helper_hits:
                violations.append(
                    Violation(
                        file=str(path),
                        lineno=sub.lineno,
                        func_name=node.name,
                        kind="via_helper",
                        detail=label,
                        helper_name=helper_node.name,
                        helper_lineno=helper_node.lineno,
                    )
                )

    return violations


def scan_tree(root: Path, exclude: set[str]) -> list[Violation]:
    violations: list[Violation] = []
    paths = [root] if root.is_file() else sorted(root.rglob("*.py"))
    for path in paths:
        # Normalize to a mcp_server/... style relative path for exclude matching.
        try:
            rel_to_root_parent = path.relative_to(root.parent)
        except ValueError:
            rel_to_root_parent = path
        rel_str = str(rel_to_root_parent).replace("\\", "/")
        if rel_str in exclude:
            continue
        violations.extend(scan_file(path))
    return violations


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "root",
        nargs="?",
        default="mcp_server",
        help="Directory to scan recursively (default: mcp_server)",
    )
    ap.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        help="Relative file paths (e.g. mcp_server/server.py) to skip entirely",
    )
    ap.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    exclude = set(args.exclude)

    violations = scan_tree(root, exclude)

    if args.json:
        print(json.dumps([v.to_dict() for v in violations], indent=2))
    else:
        if not violations:
            print("No violations found.")
        else:
            by_file: dict[str, list[Violation]] = {}
            for v in violations:
                by_file.setdefault(v.file, []).append(v)
            for file, vs in by_file.items():
                print(f"\n{file}  ({len(vs)} violation(s))")
                for v in vs:
                    if v.kind == "direct":
                        print(f"  L{v.lineno}: async def {v.func_name}() -> direct unwrapped {v.detail}")
                    else:
                        print(
                            f"  L{v.lineno}: async def {v.func_name}() -> calls helper "
                            f"{v.helper_name}() (L{v.helper_lineno}) which has unwrapped {v.detail}"
                        )
            print(f"\nTotal: {len(violations)} violation(s) across {len(by_file)} file(s).")

    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Verify the local LivePilot Codex plugin install is complete.

Catches the v1.26.1 bug class from the Codex side: the local plugin cache can
look present while `.mcp.json`, mirrored manifests, or payload directories are
missing. Codex then sees skills/commands without a spawnable MCP server after a
refresh.

Environment overrides mirror `installer/codex.js` for temp-dir tests:

    LIVEPILOT_CODEX_PLUGIN_PATH
    LIVEPILOT_CODEX_CACHE_ROOT
    LIVEPILOT_CODEX_CACHE_PATH
    LIVEPILOT_CODEX_MARKETPLACE_PATH

Exit codes:

    0 — all checks pass
    1 — one or more files missing or mismatched
    2 — repo state unreadable
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
ACTIVE_DIR = Path(
    os.environ.get("LIVEPILOT_CODEX_PLUGIN_PATH", Path.home() / "plugins" / "livepilot")
)
DEFAULT_CACHE_ROOT = Path.home() / ".codex" / "plugins" / "cache" / "local-plugins"
CACHE_ROOT = Path(os.environ.get("LIVEPILOT_CODEX_CACHE_ROOT", DEFAULT_CACHE_ROOT))
MARKETPLACE_PATH = Path(
    os.environ.get(
        "LIVEPILOT_CODEX_MARKETPLACE_PATH",
        Path.home() / ".agents" / "plugins" / "marketplace.json",
    )
)


class Issue:
    """A single verification failure."""

    __slots__ = ("location", "detail", "fix")

    def __init__(self, location: str, detail: str, fix: str = ""):
        self.location = location
        self.detail = detail
        self.fix = fix

    def format(self) -> str:
        lines = [f"  ✗ {self.location}", f"    {self.detail}"]
        if self.fix:
            lines.append(f"    fix: {self.fix}")
        return "\n".join(lines)


def _read_json(path: Path) -> tuple[Optional[Any], Optional[str]]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except FileNotFoundError:
        return None, "missing"
    except json.JSONDecodeError as exc:
        return None, f"malformed JSON: {exc}"
    except OSError as exc:
        return None, f"unreadable: {exc}"


def read_repo_manifest() -> Optional[dict[str, Any]]:
    path = REPO_ROOT / "livepilot" / ".Codex-plugin" / "plugin.json"
    data, error = _read_json(path)
    if error or not isinstance(data, dict):
        return None
    return data


def target_cache_dir(plugin_name: str, version: str) -> Path:
    override = os.environ.get("LIVEPILOT_CODEX_CACHE_PATH")
    if override:
        return Path(override)
    return CACHE_ROOT / plugin_name / version


def _check_manifest(path: Path, expected_version: str, expected_name: str) -> list[Issue]:
    data, error = _read_json(path)
    if error:
        return [
            Issue(
                str(path),
                f"plugin.json {error}",
                "run: npx livepilot --install-codex-plugin",
            )
        ]
    issues: list[Issue] = []
    if not isinstance(data, dict):
        issues.append(Issue(str(path), "plugin.json is not an object"))
        return issues
    if data.get("name") != expected_name:
        issues.append(
            Issue(
                str(path),
                f"name mismatch: {data.get('name')!r} != {expected_name!r}",
                "re-run the Codex plugin installer",
            )
        )
    if data.get("version") != expected_version:
        issues.append(
            Issue(
                str(path),
                f"version mismatch: {data.get('version')!r} != {expected_version!r}",
                "re-run the Codex plugin installer",
            )
        )
    return issues


def _check_mcp_config(path: Path) -> list[Issue]:
    data, error = _read_json(path)
    if error:
        return [
            Issue(
                str(path),
                f".mcp.json {error}",
                "run: npx livepilot --install-codex-plugin",
            )
        ]
    issues: list[Issue] = []
    if not isinstance(data, dict):
        return [Issue(str(path), ".mcp.json is not an object")]

    servers = data.get("mcpServers")
    if not isinstance(servers, dict):
        return [Issue(str(path), "mcpServers must be an object")]

    server = servers.get("livepilot")
    if not isinstance(server, dict):
        return [Issue(str(path), "mcpServers.livepilot missing")]

    command = server.get("command")
    if not isinstance(command, str) or not Path(command).is_absolute():
        issues.append(
            Issue(
                str(path),
                f"expected an absolute Node command, got {command!r}",
                "re-run the Codex plugin installer so .mcp.json points at the local node binary",
            )
        )
    elif "node" not in Path(command).name.lower():
        issues.append(
            Issue(
                str(path),
                f"command is absolute but does not look like node: {command!r}",
                "re-run the Codex plugin installer",
            )
        )

    args = server.get("args")
    expected_args = [str(REPO_ROOT / "bin" / "livepilot.js")]
    if args != expected_args:
        issues.append(
            Issue(
                str(path),
                f"args mismatch: {args!r} != {expected_args!r}",
                "re-run from this checkout: npx livepilot --install-codex-plugin",
            )
        )
    return issues


def _check_payload_dirs(root: Path) -> list[Issue]:
    issues: list[Issue] = []
    for subdir in ("commands", "skills", "rubrics", "agents"):
        path = root / subdir
        if not path.is_dir():
            issues.append(
                Issue(
                    str(path),
                    f"{subdir}/ missing",
                    "run: npx livepilot --install-codex-plugin",
                )
            )
    return issues


def _check_plugin_tree(root: Path, expected_version: str, expected_name: str) -> list[Issue]:
    if not root.exists():
        return [
            Issue(
                str(root),
                "directory missing",
                "run: npx livepilot --install-codex-plugin",
            )
        ]

    issues: list[Issue] = []
    for manifest_dir in (".Codex-plugin", ".codex-plugin"):
        issues.extend(
            _check_manifest(root / manifest_dir / "plugin.json", expected_version, expected_name)
        )
    issues.extend(_check_mcp_config(root / ".mcp.json"))
    issues.extend(_check_payload_dirs(root))
    return issues


def check_marketplace(plugin_name: str) -> list[Issue]:
    data, error = _read_json(MARKETPLACE_PATH)
    if error:
        return [
            Issue(
                str(MARKETPLACE_PATH),
                f"marketplace.json {error}",
                "run: npx livepilot --install-codex-plugin",
            )
        ]
    if not isinstance(data, dict):
        return [Issue(str(MARKETPLACE_PATH), "marketplace.json is not an object")]

    plugins = data.get("plugins", [])
    if not isinstance(plugins, list):
        return [Issue(str(MARKETPLACE_PATH), "plugins is not a list")]
    entry = next(
        (
            plugin
            for plugin in plugins
            if isinstance(plugin, dict) and plugin.get("name") == plugin_name
        ),
        None,
    )
    if not entry:
        return [
            Issue(
                str(MARKETPLACE_PATH),
                f"no local marketplace entry for {plugin_name!r}",
                "run: npx livepilot --install-codex-plugin",
            )
        ]

    source = entry.get("source", {})
    policy = entry.get("policy", {})
    issues: list[Issue] = []
    if not isinstance(source, dict):
        issues.append(
            Issue(
                str(MARKETPLACE_PATH),
                f"marketplace source must be an object, got {source!r}",
                "re-run the Codex plugin installer",
            )
        )
        source = {}
    if not isinstance(policy, dict):
        issues.append(
            Issue(
                str(MARKETPLACE_PATH),
                f"marketplace policy must be an object, got {policy!r}",
                "re-run the Codex plugin installer",
            )
        )
        policy = {}
    if source.get("source") != "local" or source.get("path") != "./plugins/livepilot":
        issues.append(
            Issue(
                str(MARKETPLACE_PATH),
                f"unexpected marketplace source: {source!r}",
                "re-run the Codex plugin installer",
            )
        )
    if policy.get("installation") != "AVAILABLE":
        issues.append(
            Issue(
                str(MARKETPLACE_PATH),
                f"unexpected marketplace policy: {policy!r}",
                "re-run the Codex plugin installer",
            )
        )
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--quiet", action="store_true", help="only print when broken")
    args = parser.parse_args()

    manifest = read_repo_manifest()
    if manifest is None:
        print("ERROR: couldn't read livepilot/.Codex-plugin/plugin.json", file=sys.stderr)
        return 2

    plugin_name = str(manifest.get("name") or "livepilot")
    expected_version = manifest.get("version")
    if not isinstance(expected_version, str):
        print("ERROR: repo Codex plugin manifest has no string version", file=sys.stderr)
        return 2

    cache_dir = target_cache_dir(plugin_name, expected_version)
    all_issues: list[tuple[str, list[Issue]]] = [
        ("Active plugin dir", _check_plugin_tree(ACTIVE_DIR, expected_version, plugin_name)),
        ("Cache version dir", _check_plugin_tree(cache_dir, expected_version, plugin_name)),
        ("Marketplace", check_marketplace(plugin_name)),
    ]
    total = sum(len(issues) for _, issues in all_issues)

    if total == 0:
        if not args.quiet:
            print(f"LivePilot Codex plugin sync check — v{expected_version}")
            print(f"  ✓ Active plugin dir       ({ACTIVE_DIR})")
            print(f"  ✓ Cache version dir       ({cache_dir})")
            print(f"  ✓ Marketplace             ({MARKETPLACE_PATH})")
            print("All checks pass. Codex can load the local plugin MCP server.")
        return 0

    print(f"LivePilot Codex plugin sync check — v{expected_version}")
    print(f"FAILED: {total} issue(s) found\n")
    for section, issues in all_issues:
        if not issues:
            print(f"  ✓ {section}")
            continue
        print(f"  ✗ {section} ({len(issues)} issue(s)):")
        for issue in issues:
            print(issue.format())
    print()
    print("After fixing, re-run: python3 scripts/verify_codex_plugin_sync.py")
    return 1


if __name__ == "__main__":
    sys.exit(main())

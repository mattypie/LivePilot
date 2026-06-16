"""Regression tests for the Codex local-plugin sync verifier."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "verify_codex_plugin_sync.py"


def _expected_manifest() -> dict:
    return json.loads(
        (REPO_ROOT / "livepilot" / ".Codex-plugin" / "plugin.json").read_text(
            encoding="utf-8"
        )
    )


def _write_plugin_tree(path: Path, manifest: dict) -> None:
    for manifest_dir in (".Codex-plugin", ".codex-plugin"):
        target = path / manifest_dir
        target.mkdir(parents=True, exist_ok=True)
        (target / "plugin.json").write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )

    (path / ".mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "livepilot": {
                        "command": "/usr/local/bin/node",
                        "args": [str(REPO_ROOT / "bin" / "livepilot.js")],
                    }
                }
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    for subdir in ("commands", "skills", "rubrics", "agents"):
        child = path / subdir
        child.mkdir(parents=True, exist_ok=True)
        (child / ".keep").write_text("", encoding="utf-8")


def _write_marketplace(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "name": "local-plugins",
                "interface": {"displayName": "Local Plugins"},
                "plugins": [
                    {
                        "name": "livepilot",
                        "source": {"source": "local", "path": "./plugins/livepilot"},
                        "policy": {
                            "installation": "AVAILABLE",
                            "authentication": "ON_INSTALL",
                        },
                    }
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _env(tmp_path: Path, active_dir: Path, cache_dir: Path, marketplace: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["LIVEPILOT_CODEX_PLUGIN_PATH"] = str(active_dir)
    env["LIVEPILOT_CODEX_CACHE_PATH"] = str(cache_dir)
    env["LIVEPILOT_CODEX_MARKETPLACE_PATH"] = str(marketplace)
    env["HOME"] = str(tmp_path)
    return env


def _run_verifier(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--quiet"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )


def test_codex_verifier_passes_complete_temp_install(tmp_path: Path):
    manifest = _expected_manifest()
    active_dir = tmp_path / "plugins" / "livepilot"
    cache_dir = tmp_path / ".codex" / "plugins" / "cache" / "local-plugins" / "livepilot" / manifest["version"]
    marketplace = tmp_path / ".agents" / "plugins" / "marketplace.json"

    _write_plugin_tree(active_dir, manifest)
    _write_plugin_tree(cache_dir, manifest)
    _write_marketplace(marketplace)

    result = _run_verifier(_env(tmp_path, active_dir, cache_dir, marketplace))
    assert result.returncode == 0, result.stdout + result.stderr


def test_codex_verifier_fails_when_cache_mcp_missing(tmp_path: Path):
    manifest = _expected_manifest()
    active_dir = tmp_path / "plugins" / "livepilot"
    cache_dir = tmp_path / ".codex" / "plugins" / "cache" / "local-plugins" / "livepilot" / manifest["version"]
    marketplace = tmp_path / ".agents" / "plugins" / "marketplace.json"

    _write_plugin_tree(active_dir, manifest)
    _write_plugin_tree(cache_dir, manifest)
    (cache_dir / ".mcp.json").unlink()
    _write_marketplace(marketplace)

    result = _run_verifier(_env(tmp_path, active_dir, cache_dir, marketplace))
    assert result.returncode == 1
    assert ".mcp.json missing" in result.stdout


def test_codex_verifier_rejects_repo_npx_mcp_config_in_cache(tmp_path: Path):
    manifest = _expected_manifest()
    active_dir = tmp_path / "plugins" / "livepilot"
    cache_dir = tmp_path / ".codex" / "plugins" / "cache" / "local-plugins" / "livepilot" / manifest["version"]
    marketplace = tmp_path / ".agents" / "plugins" / "marketplace.json"

    _write_plugin_tree(active_dir, manifest)
    _write_plugin_tree(cache_dir, manifest)
    (cache_dir / ".mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "livepilot": {"command": "npx", "args": ["livepilot"]}
                }
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_marketplace(marketplace)

    result = _run_verifier(_env(tmp_path, active_dir, cache_dir, marketplace))
    assert result.returncode == 1
    assert "absolute Node command" in result.stdout


def test_codex_verifier_reports_malformed_mcp_shape(tmp_path: Path):
    manifest = _expected_manifest()
    active_dir = tmp_path / "plugins" / "livepilot"
    cache_dir = tmp_path / ".codex" / "plugins" / "cache" / "local-plugins" / "livepilot" / manifest["version"]
    marketplace = tmp_path / ".agents" / "plugins" / "marketplace.json"

    _write_plugin_tree(active_dir, manifest)
    _write_plugin_tree(cache_dir, manifest)
    (cache_dir / ".mcp.json").write_text(
        json.dumps({"mcpServers": []}, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_marketplace(marketplace)

    result = _run_verifier(_env(tmp_path, active_dir, cache_dir, marketplace))
    assert result.returncode == 1
    assert "mcpServers must be an object" in result.stdout

"""Tests for the Remote Script installer (installer/install.js).

Mirrors the subprocess pattern used by test_codex_plugin_installer.py:
invoke `node` against a sandboxed HOME so the real installer code path
(candidate auto-detection, clear-then-copy upgrade, error handling) runs
unmodified against a fake Ableton directory tree under tmp_path.

No env-var override needed for candidate detection: installer/paths.js
already derives every candidate from `os.homedir()`, and Node's
`os.homedir()` honors the `HOME` env var on POSIX. Pointing a subprocess's
HOME at a tmp_path sandbox is therefore sufficient to redirect
`findAbletonPaths()` (and the installer's own safe-path check, which
requires the target to live under `os.homedir()`) without touching
installer source.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest


NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="node not available")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _fake_home(tmp_path: Path) -> tuple[Path, Path]:
    """Create a fake HOME with the default macOS/Windows-style candidate dir.

    Returns (home_dir, remote_scripts_dir).
    """
    home = tmp_path / "home"
    remote_scripts = home / "Music" / "Ableton" / "User Library" / "Remote Scripts"
    remote_scripts.mkdir(parents=True)
    return home, remote_scripts


def _env_for_home(home: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["HOME"] = str(home)
    # Make sure a stray real install path never leaks into the sandbox.
    env.pop("LIVEPILOT_INSTALL_PATH", None)
    return env


def _run_install(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [NODE, "-e", "require('./installer/install.js').install()"],
        cwd=_repo_root(),
        env=env,
        text=True,
        capture_output=True,
    )


def test_install_places_remote_script_under_user_library(tmp_path: Path):
    home, remote_scripts = _fake_home(tmp_path)
    env = _env_for_home(home)

    result = _run_install(env)
    assert result.returncode == 0, result.stderr

    dest = remote_scripts / "LivePilot"
    assert dest.is_dir()
    assert (dest / "__init__.py").is_file()

    # Content should match the repo's source file byte-for-byte.
    src_init = _repo_root() / "remote_script" / "LivePilot" / "__init__.py"
    assert (dest / "__init__.py").read_text(encoding="utf-8") == src_init.read_text(encoding="utf-8")


def test_install_overwrites_existing_install_cleanly(tmp_path: Path):
    home, remote_scripts = _fake_home(tmp_path)
    env = _env_for_home(home)

    first = _run_install(env)
    assert first.returncode == 0, first.stderr

    dest = remote_scripts / "LivePilot"
    stale_file = dest / "stale_leftover_module.py"
    stale_file.write_text("# should not survive a reinstall\n", encoding="utf-8")
    assert stale_file.exists()

    second = _run_install(env)
    assert second.returncode == 0, second.stderr

    # Clear-then-copy: the stale file must be gone after reinstall, and the
    # fresh source file set must still be present.
    assert not stale_file.exists()
    assert (dest / "__init__.py").is_file()

    # A timestamped backup of the previous install must have been created
    # (the installer renames, not deletes, the prior LivePilot/ dir).
    backups = [p for p in remote_scripts.iterdir() if p.name.startswith("LivePilot.backup-")]
    assert len(backups) == 1
    assert (backups[0] / "stale_leftover_module.py").exists()


def test_install_missing_ableton_dir_yields_clear_error_not_stack_trace(tmp_path: Path):
    # Empty HOME: no Music/Ableton tree, no Library/Preferences/Ableton tree.
    # findAbletonPaths() returns [] and install() throws a recoverable
    # InstallerAbort with actionable manual-install instructions.
    home = tmp_path / "empty_home"
    home.mkdir()
    env = _env_for_home(home)

    script = (
        "const { install, InstallerAbort } = require('./installer/install.js');"
        "try { install(); process.exit(0); }"
        "catch (e) {"
        "  if (e instanceof InstallerAbort) {"
        "    console.error(e.message);"
        "    process.exit(e.recoverable ? 2 : 1);"
        "  }"
        "  throw e;"
        "}"
    )
    result = subprocess.run(
        [NODE, "-e", script],
        cwd=_repo_root(),
        env=env,
        text=True,
        capture_output=True,
    )

    # Recoverable InstallerAbort -> exit code 2 (mirrors bin/livepilot.js's
    # own --install error handling), not an uncaught-exception exit code.
    assert result.returncode == 2, result.stdout + result.stderr
    assert "Could not auto-detect an Ableton Live Remote Scripts directory" in result.stderr
    assert "Manual install" in result.stderr

    # A raw Node stack trace looks like "    at Object.<anonymous> (...)" —
    # assert we printed a clean message instead of letting one leak through.
    assert "    at " not in result.stderr


def test_install_path_traversal_outside_home_is_refused(tmp_path: Path):
    """LIVEPILOT_INSTALL_PATH pointed outside HOME/known Ableton roots must
    be refused with InstallerAbort, not silently followed."""
    home, _remote_scripts = _fake_home(tmp_path)
    outside = tmp_path / "definitely_not_home"
    outside.mkdir()

    env = _env_for_home(home)
    env["LIVEPILOT_INSTALL_PATH"] = str(outside)

    script = (
        "const { install, InstallerAbort } = require('./installer/install.js');"
        "try { install(); process.exit(0); }"
        "catch (e) {"
        "  if (e instanceof InstallerAbort) { console.error(e.message); process.exit(1); }"
        "  throw e;"
        "}"
    )
    result = subprocess.run(
        [NODE, "-e", script],
        cwd=_repo_root(),
        env=env,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "outside permitted directories" in result.stderr
    assert not (outside / "LivePilot").exists()

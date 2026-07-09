"""Regression tests for store schema-versioning/migration and thread capping.

Covers the state-layer hardening pass:
  A. Version mismatch on either persisted store must back up the raw file
     (never silently discard), log a WARNING, and fall back to defaults.
     A registered migration path must still be honored (no unnecessary
     backup/reset when the store knows how to upgrade the data).
  B. ProjectStore.save_thread must cap at _MAX_THREADS, pruning resolved
     threads (oldest-touched first) before ever dropping an open thread.
"""

import json
import logging
import tempfile
from pathlib import Path

import mcp_server.persistence.project_store as project_store_mod
import mcp_server.persistence.taste_store as taste_store_mod
from mcp_server.persistence.project_store import ProjectStore, _MAX_THREADS
from mcp_server.persistence.taste_store import PersistentTasteStore


# ── Taste store: version mismatch → backup + warn + defaults ────────


def test_taste_store_unrecognized_version_backs_up_and_warns(caplog):
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "taste.json"
        path.write_text(json.dumps({
            "version": 999,
            "move_outcomes": {"make_punchier": {"kept_count": 7}},
        }))

        store = PersistentTasteStore(path)
        with caplog.at_level(logging.WARNING):
            data = store.get_all()

        # Falls back to defaults — does NOT surface the stale v999 payload.
        assert data["version"] == 1
        assert data["move_outcomes"] == {}

        # The raw file was backed up, not discarded.
        backup = path.with_suffix(path.suffix + ".pre-migration")
        assert backup.exists()
        backed_up = json.loads(backup.read_text())
        assert backed_up["move_outcomes"]["make_punchier"]["kept_count"] == 7

        assert any(
            "unrecognized schema version" in r.getMessage().lower()
            for r in caplog.records
        ), "expected a WARNING log on version mismatch"


def test_taste_store_unrecognized_version_via_write_path_also_backs_up(caplog):
    """The same guard must apply on the read-modify-write path (record_*),
    not just get_all() — every mutator routes through _coerce()."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "taste.json"
        path.write_text(json.dumps({"version": 2, "evidence_count": 41}))

        store = PersistentTasteStore(path)
        with caplog.at_level(logging.WARNING):
            store.record_move_outcome("widen_stereo", "mix", kept=True)

        backup = path.with_suffix(path.suffix + ".pre-migration")
        assert backup.exists()
        assert json.loads(backup.read_text())["evidence_count"] == 41

        # The write proceeded against a fresh default, not the stale payload.
        data = store.get_all()
        assert data["move_outcomes"]["widen_stereo"]["kept_count"] == 1
        assert data["evidence_count"] == 1


def test_taste_store_missing_file_returns_defaults_without_backup(caplog):
    """No on-disk file at all is not an 'unrecognized version' event —
    there's nothing to back up, and no warning should fire."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "does_not_exist.json"
        store = PersistentTasteStore(path)
        with caplog.at_level(logging.WARNING):
            data = store.get_all()

        assert data["version"] == 1
        backup = path.with_suffix(path.suffix + ".pre-migration")
        assert not backup.exists()
        assert not any(
            "unrecognized schema version" in r.getMessage().lower()
            for r in caplog.records
        )


def test_taste_store_registered_migration_runs_without_backup(monkeypatch):
    """When a migration path IS registered for the on-disk version, it must
    run instead of falling back to defaults — proving the scaffolding
    actually upgrades data rather than just gating on CURRENT_VERSION."""

    def _migrate_v0_to_v1(data: dict) -> dict:
        data = dict(data)
        data["version"] = 1
        data["migrated_from_v0"] = True
        data.setdefault("move_outcomes", {})
        return data

    monkeypatch.setitem(taste_store_mod._MIGRATIONS, 0, _migrate_v0_to_v1)

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "taste.json"
        path.write_text(json.dumps({"version": 0, "move_outcomes": {"x": {"kept_count": 3}}}))

        store = PersistentTasteStore(path)
        data = store.get_all()

        assert data["version"] == 1
        assert data.get("migrated_from_v0") is True
        assert data["move_outcomes"]["x"]["kept_count"] == 3

        backup = path.with_suffix(path.suffix + ".pre-migration")
        assert not backup.exists(), "a successful migration must not trigger a backup"


# ── Project store: version mismatch → backup + warn + defaults ──────


def test_project_store_unrecognized_version_backs_up_and_warns(caplog):
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        path = base / "proj1" / "state.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({
            "version": 7,
            "threads": [{"thread_id": "old", "status": "open"}],
        }))

        store = ProjectStore("proj1", base_dir=base)
        with caplog.at_level(logging.WARNING):
            data = store.get_all()

        assert data["version"] == 1
        assert data["threads"] == []

        backup = path.with_suffix(path.suffix + ".pre-migration")
        assert backup.exists()
        assert json.loads(backup.read_text())["threads"][0]["thread_id"] == "old"

        assert any(
            "unrecognized schema version" in r.getMessage().lower()
            for r in caplog.records
        )


def test_project_store_registered_migration_runs_without_backup(monkeypatch):
    def _migrate_v0_to_v1(data: dict) -> dict:
        data = dict(data)
        data["version"] = 1
        data["migrated_from_v0"] = True
        return data

    monkeypatch.setitem(project_store_mod._MIGRATIONS, 0, _migrate_v0_to_v1)

    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        path = base / "proj1" / "state.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"version": 0, "threads": [{"thread_id": "t1"}]}))

        store = ProjectStore("proj1", base_dir=base)
        data = store.get_all()

        assert data["version"] == 1
        assert data.get("migrated_from_v0") is True
        assert data["threads"][0]["thread_id"] == "t1"

        backup = path.with_suffix(path.suffix + ".pre-migration")
        assert not backup.exists()


# ── Thread capping ────────────────────────────────────────────────


def test_thread_cap_prunes_oldest_resolved_first():
    with tempfile.TemporaryDirectory() as d:
        store = ProjectStore("proj1", base_dir=Path(d))

        for i in range(_MAX_THREADS):
            store.save_thread({
                "thread_id": f"resolved_{i}",
                "description": f"r{i}",
                "status": "resolved",
                "last_touched_ms": i,  # ascending -> resolved_0 is oldest
            })

        # One more (open) thread pushes the list 1 over the cap.
        store.save_thread({
            "thread_id": "open_new",
            "description": "new open thread",
            "status": "open",
            "last_touched_ms": _MAX_THREADS + 1000,
        })

        threads = ProjectStore("proj1", base_dir=Path(d)).get_threads()
        assert len(threads) == _MAX_THREADS
        ids = {t["thread_id"] for t in threads}
        assert "open_new" in ids, "the new open thread must survive"
        assert "resolved_0" not in ids, "oldest resolved thread should be pruned first"
        assert "resolved_1" in ids, "only the overflow amount should be pruned"


def test_thread_cap_falls_back_to_oldest_touched_when_all_open():
    """When there are no resolved threads to sacrifice, the oldest-touched
    threads (regardless of status) are dropped so recency wins."""
    with tempfile.TemporaryDirectory() as d:
        store = ProjectStore("proj1", base_dir=Path(d))

        for i in range(_MAX_THREADS + 5):
            store.save_thread({
                "thread_id": f"open_{i}",
                "description": f"o{i}",
                "status": "open",
                "last_touched_ms": i,
            })

        threads = ProjectStore("proj1", base_dir=Path(d)).get_threads()
        assert len(threads) == _MAX_THREADS
        ids = {t["thread_id"] for t in threads}
        for i in range(5):
            assert f"open_{i}" not in ids, "oldest-touched open threads should be pruned"
        assert f"open_{_MAX_THREADS + 4}" in ids, "most-recently-touched thread must survive"


def test_thread_cap_not_triggered_below_limit():
    """Sanity: normal usage well under the cap is untouched."""
    with tempfile.TemporaryDirectory() as d:
        store = ProjectStore("proj1", base_dir=Path(d))
        for i in range(5):
            store.save_thread({"thread_id": f"t{i}", "status": "open", "last_touched_ms": i})
        threads = ProjectStore("proj1", base_dir=Path(d)).get_threads()
        assert len(threads) == 5

"""Tests for the security-hardening batch:

1. corpus_trim_plugin_identity path-traversal / arbitrary-plugin_id rejection
2. SpectralReceiver UDP source-address allowlist (loopback-only)
3. Client-eviction error message always surfaces the identified peer
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mcp_server.connection import AbletonConnection, AbletonConnectionError
import mcp_server.connection as connection_mod
from mcp_server.m4l_bridge import SpectralCache, SpectralReceiver
from mcp_server.user_corpus import tools as corpus_tools


# ---------------------------------------------------------------------------
# 1. corpus_trim_plugin_identity
# ---------------------------------------------------------------------------


class _FakeContext:
    """Minimal stand-in for fastmcp.Context; the tool doesn't use it directly."""


def _write_inventory(root, plugin_ids):
    plugins_dir = root / "plugins"
    plugins_dir.mkdir(parents=True, exist_ok=True)
    inventory = {
        "plugins": [
            {"plugin_id": pid, "vendor": "acme", "name": pid} for pid in plugin_ids
        ]
    }
    (plugins_dir / "_inventory.json").write_text(json.dumps(inventory), encoding="utf-8")
    return plugins_dir


def _write_identity_yaml(plugins_dir, plugin_id):
    import yaml as _yaml

    plugin_dir = plugins_dir / plugin_id
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "identity.yaml").write_text(
        _yaml.dump({
            "entity_id": plugin_id,
            "entity_type": "installed_plugin",
            "name": plugin_id,
            "description": "desc",
            "tags": ["genre:house"],
            "vendor": "acme",
            "format": "vst3",
        }),
        encoding="utf-8",
    )


def test_trim_plugin_identity_rejects_path_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr(corpus_tools, "DEFAULT_OUTPUT_ROOT", tmp_path)
    _write_inventory(tmp_path, ["acme-synth"])

    traversal_id = "../../etc/passwd"
    result = corpus_tools.corpus_trim_plugin_identity(
        _FakeContext(), plugin_id=traversal_id
    )

    assert result["status"] == "error"
    assert result["code"] == "INVALID_PARAM"
    # Must not have created/traversed anything outside the plugins root.
    assert not (tmp_path / "plugins" / traversal_id).exists()


def test_trim_plugin_identity_rejects_id_not_in_inventory(tmp_path, monkeypatch):
    monkeypatch.setattr(corpus_tools, "DEFAULT_OUTPUT_ROOT", tmp_path)
    _write_inventory(tmp_path, ["acme-synth"])

    result = corpus_tools.corpus_trim_plugin_identity(
        _FakeContext(), plugin_id="not-in-inventory"
    )

    assert result["status"] == "error"
    assert result["code"] == "INVALID_PARAM"


def test_trim_plugin_identity_accepts_valid_slug(tmp_path, monkeypatch):
    monkeypatch.setattr(corpus_tools, "DEFAULT_OUTPUT_ROOT", tmp_path)
    plugins_dir = _write_inventory(tmp_path, ["acme-synth"])
    _write_identity_yaml(plugins_dir, "acme-synth")

    result = corpus_tools.corpus_trim_plugin_identity(
        _FakeContext(), plugin_id="acme-synth", research_priority="low"
    )

    assert "error" not in result
    assert result["plugin_id"] == "acme-synth"


# ---------------------------------------------------------------------------
# 2. SpectralReceiver UDP source-address allowlist
# ---------------------------------------------------------------------------


def _spectrum_packet():
    # Minimal OSC-ish payload; SpectralReceiver only needs to reach
    # _parse_osc without raising for this test, actual band parsing is
    # covered elsewhere (test_m4l_bridge_band_names.py).
    return b"/peak\x00\x00\x00,f\x00\x00\x00\x00\x00\x00"


def test_datagram_from_non_loopback_is_dropped(monkeypatch):
    cache = SpectralCache()
    receiver = SpectralReceiver(cache)

    calls = []
    monkeypatch.setattr(receiver, "_parse_osc", lambda data: calls.append(data))

    receiver.datagram_received(_spectrum_packet(), ("10.0.0.5", 54321))

    assert calls == []


def test_datagram_from_loopback_is_accepted(monkeypatch):
    cache = SpectralCache()
    receiver = SpectralReceiver(cache)

    calls = []
    monkeypatch.setattr(receiver, "_parse_osc", lambda data: calls.append(data))

    receiver.datagram_received(_spectrum_packet(), ("127.0.0.1", 54321))

    assert len(calls) == 1


def test_datagram_from_ipv6_loopback_is_accepted(monkeypatch):
    cache = SpectralCache()
    receiver = SpectralReceiver(cache)

    calls = []
    monkeypatch.setattr(receiver, "_parse_osc", lambda data: calls.append(data))

    receiver.datagram_received(_spectrum_packet(), ("::1", 54321))

    assert len(calls) == 1


def test_non_loopback_warning_is_throttled_per_source(monkeypatch, capsys):
    cache = SpectralCache()
    receiver = SpectralReceiver(cache)
    monkeypatch.setattr(receiver, "_parse_osc", lambda data: None)

    for _ in range(5):
        receiver.datagram_received(_spectrum_packet(), ("10.0.0.5", 54321))

    err = capsys.readouterr().err
    assert err.count("dropping UDP packet") == 1


# ---------------------------------------------------------------------------
# 3. Client-eviction error surfaces peer info
# ---------------------------------------------------------------------------


def test_single_client_error_includes_peer_when_retry_also_fails(monkeypatch):
    conn = AbletonConnection(host="127.0.0.1", port=19999)

    def fake_connect():
        conn._socket = object()

    def fake_disconnect():
        conn._socket = None

    def fake_send_raw(command, recv_timeout=20):
        return {
            "ok": False,
            "error": {
                "code": "STATE_ERROR",
                "message": "Another client is already connected. LivePilot accepts one client at a time. Disconnect the current client first.",
            },
        }

    monkeypatch.setattr(conn, "connect", fake_connect)
    monkeypatch.setattr(conn, "disconnect", fake_disconnect)
    monkeypatch.setattr(conn, "_send_raw", fake_send_raw)
    monkeypatch.setattr(connection_mod.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        connection_mod, "_identify_other_tcp_client", lambda host, port: "PID 4242 (node)"
    )

    with pytest.raises(AbletonConnectionError, match=r"PID 4242 \(node\)"):
        conn.send_command("ping")


def test_single_client_error_without_resolvable_peer_still_reports_state_error(monkeypatch):
    conn = AbletonConnection(host="127.0.0.1", port=19999)

    def fake_connect():
        conn._socket = object()

    def fake_disconnect():
        conn._socket = None

    def fake_send_raw(command, recv_timeout=20):
        return {
            "ok": False,
            "error": {
                "code": "STATE_ERROR",
                "message": "Another client is already connected. LivePilot accepts one client at a time. Disconnect the current client first.",
            },
        }

    monkeypatch.setattr(conn, "connect", fake_connect)
    monkeypatch.setattr(conn, "disconnect", fake_disconnect)
    monkeypatch.setattr(conn, "_send_raw", fake_send_raw)
    monkeypatch.setattr(connection_mod.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(connection_mod, "_identify_other_tcp_client", lambda host, port: None)

    with pytest.raises(AbletonConnectionError, match="Another client is already connected"):
        conn.send_command("ping")

"""End-to-end tests driving real MCP tools through FastMCP's dispatch layer.

Every other test in this suite calls tool functions directly (bypassing
schema validation, argument coercion, and the lifespan-injected Context).
Zero tests previously drove a tool through ``mcp.call_tool`` / the
in-process fastmcp.Client — meaning the actual dispatch path a real MCP
client uses (JSON args -> pydantic schema validation/coercion -> the
lifespan-scoped ``ctx.lifespan_context["ableton"]`` -> the tool body) had
no coverage at all.

fastmcp 3.x exposes an in-memory client (``fastmcp.Client(app)``) that runs
the full lifespan + middleware + schema-validation stack without opening a
real transport (no stdio/HTTP socket). We drive `mcp.call_tool` this way,
with ``AbletonConnection.send_command``/``send_command_async`` patched at
the class level (see tests/conftest.py::mcp_client_factory) so no live
Ableton Live instance is required.

Covers:
  - a read tool (get_track_info)
  - a write tool with string->int/float argument coercion
    (set_track_volume called with string args, as a JSON-RPC client would
    send if the caller didn't type-convert)
  - a tool call that surfaces a structured Remote-Script error
    ([NOT_FOUND]/[STATE_ERROR]-style AbletonConnectionError) as a ToolError
  - an unknown-argument call, rejected by schema validation before the
    tool body ever runs
"""

from __future__ import annotations

import pytest
from fastmcp.exceptions import ToolError

from mcp_server.connection import AbletonConnectionError
from tests.fixtures_remote import make_track_info


@pytest.mark.asyncio
async def test_read_tool_get_track_info_returns_fixture_shape(mcp_client_factory):
    async with mcp_client_factory({
        "get_track_info": lambda p: make_track_info(p["track_index"], name="Bass"),
    }) as client:
        result = await client.call_tool("get_track_info", {"track_index": 2})

        assert result.data["index"] == 2
        assert result.data["name"] == "Bass"
        assert "mixer" in result.data
        assert "volume" in result.data["mixer"]


@pytest.mark.asyncio
async def test_write_tool_coerces_string_arguments(mcp_client_factory):
    """A JSON-RPC caller sending string-typed numbers (e.g. "0" / "0.5")
    for int/float-annotated tool params must be coerced by the pydantic
    schema before the tool body runs — set_track_volume's track_index:int
    and volume:float parameters should both accept string input here.
    """
    async with mcp_client_factory({
        "set_track_volume": lambda p: {"index": p["track_index"], "volume": p["volume"]},
    }) as client:
        result = await client.call_tool(
            "set_track_volume", {"track_index": "0", "volume": "0.65"},
        )

        assert result.data["index"] == 0
        assert abs(result.data["volume"] - 0.65) < 1e-9

        # The fake recorded the coerced (not string) params.
        calls = [c for c in client.fake_ableton.calls if c[0] == "set_track_volume"]
        assert len(calls) == 1
        _, params = calls[0]
        assert params["track_index"] == 0
        assert isinstance(params["track_index"], int)
        assert isinstance(params["volume"], float)


@pytest.mark.asyncio
async def test_tool_surfaces_structured_remote_script_error(mcp_client_factory):
    """A Remote-Script-side structured error ({"ok": false, "error": {code,
    message}}) is raised by AbletonConnection.send_command as an
    AbletonConnectionError embedding the error code — the tool call must
    propagate that as a ToolError with the code visible in the message,
    not swallow it into a generic response.
    """
    def _raise_not_found(params):
        raise AbletonConnectionError(
            "[NOT_FOUND] Track index 7 does not exist "
            "(while running 'get_track_info')"
        )

    async with mcp_client_factory({"get_track_info": _raise_not_found}) as client:
        with pytest.raises(ToolError) as exc_info:
            await client.call_tool("get_track_info", {"track_index": 7})

        assert "NOT_FOUND" in str(exc_info.value)
        assert "Track index 7 does not exist" in str(exc_info.value)


@pytest.mark.asyncio
async def test_tool_rejects_unknown_argument(mcp_client_factory):
    """An argument that isn't part of the tool's declared schema must be
    rejected by validation before the tool body (and thus before any
    Remote Script command) ever runs.
    """
    async with mcp_client_factory() as client:
        with pytest.raises(ToolError) as exc_info:
            await client.call_tool(
                "get_track_info", {"track_index": 0, "bogus_arg": 1},
            )

        assert "bogus_arg" in str(exc_info.value) or "Unexpected" in str(exc_info.value)

        # The unrecognized-argument call must never have reached the fake —
        # validation happens strictly before the tool body executes.
        calls = [c for c in client.fake_ableton.calls if c[0] == "get_track_info"]
        assert calls == []


@pytest.mark.asyncio
async def test_tool_rejects_missing_required_argument(mcp_client_factory):
    """track_index is required (no default) — omitting it must fail
    schema validation rather than crash inside the tool body with a
    confusing KeyError/TypeError.
    """
    async with mcp_client_factory() as client:
        with pytest.raises(ToolError):
            await client.call_tool("get_track_info", {})


@pytest.mark.asyncio
async def test_batch_write_tool_reports_partial_failure_through_dispatch(mcp_client_factory):
    """batch_set_parameters' partial-success contract (top-level ok=False,
    per-entry ok True/False) must survive the full mcp.call_tool trip
    unchanged — the tool must not raise just because one of N entries
    failed application-side.
    """
    from tests.fixtures_remote import (
        make_batch_param_failure,
        make_batch_param_success,
        make_batch_set_parameters_result,
    )

    partial = make_batch_set_parameters_result([
        make_batch_param_success("Cutoff", 0.6),
        make_batch_param_failure("DoesNotExist", "Parameter 'DoesNotExist' not found"),
    ])

    async with mcp_client_factory({"batch_set_parameters": partial}) as client:
        result = await client.call_tool("batch_set_parameters", {
            "track_index": 0,
            "device_index": 0,
            "parameters": [
                {"name_or_index": "Cutoff", "value": 0.6},
                {"name_or_index": "DoesNotExist", "value": 0.3},
            ],
        })

        assert result.data["ok"] is False
        assert result.data["applied"] == 1
        assert result.data["failed"] == 1
        assert len(result.data["parameters"]) == 2

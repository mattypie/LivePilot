"""Contract tests for Device Forge MCP tools."""

from __future__ import annotations

import asyncio

import pytest


def _get_tool_names():
    from mcp_server.server import mcp
    tools = asyncio.run(mcp.list_tools())
    return {tool.name for tool in tools}


def test_device_forge_tools_registered():
    names = _get_tool_names()
    expected = {
        "generate_m4l_effect",
        "list_genexpr_templates",
        "install_m4l_device",
    }
    missing = expected - names
    assert not missing, f"Missing device forge tools: {missing}"


def test_generate_m4l_effect_param_missing_name_returns_structured_error():
    """A param dict without a 'name' key must return INVALID_PARAM, not raise
    a bare KeyError out of the async tool."""
    from mcp_server.device_forge.tools import generate_m4l_effect

    result = asyncio.run(generate_m4l_effect(
        ctx=object(),
        name="Test",
        gen_code="out1 = in1;",
        device_type="audio_effect",
        params=[{"default": 0.5}],  # no "name"
        install=False,
    ))
    assert "error" in result and "INVALID_PARAM" in result["error"]


def test_generate_m4l_effect_midi_generator_builds_without_crash():
    """midi_generator is an advertised type; it must build, not KeyError."""
    from mcp_server.device_forge.tools import generate_m4l_effect

    result = asyncio.run(generate_m4l_effect(
        ctx=object(),
        name="MidiGen",
        gen_code="",
        device_type="midi_generator",
        install=False,
    ))
    assert result.get("status") == "created", result

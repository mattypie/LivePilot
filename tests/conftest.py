"""Shared pytest fixtures for tests/.

Currently just the in-process FastMCP client harness used by
test_mcp_e2e.py to drive real tools through mcp.call_tool() /
fastmcp.Client dispatch (schema validation, coercion, lifespan context)
without a live Ableton Live instance or a real TCP/socket server.

Kept intentionally tiny — this is not a global fixture dumping ground for
the 83 existing hand-rolled-fake test files (out of scope for this batch).
"""

from __future__ import annotations

from typing import AsyncIterator, Callable, Optional
from unittest.mock import patch

import pytest
import pytest_asyncio

from fastmcp import Client

from mcp_server.connection import AbletonConnection
from tests.fixtures_remote import make_fake_ableton


@pytest_asyncio.fixture
async def mcp_client_factory() -> AsyncIterator[Callable]:
    """Yield a factory that opens an in-process fastmcp.Client against the
    real `mcp` app, with AbletonConnection.send_command/send_command_async
    patched at the class level to a FakeAbletonConnection's methods.

    Patching at the class level (not swapping the lifespan-context object)
    is necessary because mcp_server.server.lifespan() constructs its own
    ``AbletonConnection()`` instance internally — there is no seam to inject
    a different instance without either patching the class or monkeypatching
    the lifespan function itself. Class-level patching is the smaller,
    more surgical change and matches how test_bugfixes_*.py already patches
    AbletonConnection elsewhere in this suite.

    Usage:
        async def test_x(mcp_client_factory):
            async with mcp_client_factory({"get_track_info": ...}) as client:
                result = await client.call_tool("get_track_info", {...})
    """
    from mcp_server.server import mcp

    def _factory(overrides: Optional[dict] = None):
        fake = make_fake_ableton(overrides)
        patcher_sync = patch.object(AbletonConnection, "send_command", fake.send_command)
        patcher_async = patch.object(AbletonConnection, "send_command_async", fake.send_command_async)

        class _ClientContext:
            async def __aenter__(self):
                patcher_sync.start()
                patcher_async.start()
                self._client_cm = Client(mcp)
                client = await self._client_cm.__aenter__()
                client.fake_ableton = fake  # expose call log to the test
                return client

            async def __aexit__(self, exc_type, exc, tb):
                try:
                    await self._client_cm.__aexit__(exc_type, exc, tb)
                finally:
                    patcher_async.stop()
                    patcher_sync.stop()

        return _ClientContext()

    yield _factory

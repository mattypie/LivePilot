"""Guard rail on the total MCP ``tools/list`` wire payload.

Every agent session pays the full ``tools/list`` response before its first
tool call — currently ~467 tools. Tool *descriptions* make up a large
share of that payload (long band tables, historical bug narratives,
multi-example JSON blocks accumulate silently, one docstring at a time).

This test sums the serialized wire form (name + description + input
schema) of every registered tool and fails if the total exceeds a
budget ceiling. The ceiling is set ~10% above the payload size as of the
2026-07 docstring trim pass (see CHANGELOG / the trim commit for the
before/after numbers) — enough headroom for normal growth (new tools,
legitimately expanded schemas) without silently re-accumulating the kind
of bloat that pass removed.

If this test fails because of deliberate, reviewed growth (a new domain
of tools, a schema that genuinely needs the extra fields), raise
``_TOTAL_PAYLOAD_BUDGET_BYTES`` below to a new value ~10% above the new
measured total — do not raise it "to make CI pass" without checking
whether the growth is a long prose block that belongs in a skill
reference instead (see ``livepilot/skills/livepilot-core/references/``
for the pattern: keep the operational contract in the docstring, move
historical/narrative bulk to a reference file with a one-line pointer).
"""

from __future__ import annotations

import json


# ~10% headroom above the total measured immediately after the 2026-07
# docstring trim pass (467 tools). Raise deliberately; see module docstring.
_TOTAL_PAYLOAD_BUDGET_BYTES = 368_000


def _serialized_tool_size(tool) -> int:
    payload = {
        "name": tool.name,
        "description": getattr(tool, "description", "") or "",
        "inputSchema": getattr(tool, "parameters", {}) or {},
    }
    return len(json.dumps(payload))


def test_total_tool_payload_under_budget():
    from mcp_server.server import _get_all_tools

    tools = _get_all_tools()
    assert tools, "Tool registry returned no tools — probe likely broken"

    total_bytes = sum(_serialized_tool_size(t) for t in tools)

    assert total_bytes <= _TOTAL_PAYLOAD_BUDGET_BYTES, (
        f"Total serialized tool payload is {total_bytes} bytes across "
        f"{len(tools)} tools — over the {_TOTAL_PAYLOAD_BUDGET_BYTES} byte "
        "budget (~10% above the post-trim baseline). This payload is paid "
        "by every agent session before its first tool call, so growth here "
        "is expensive at scale. If the growth is deliberate and reviewed, "
        "raise _TOTAL_PAYLOAD_BUDGET_BYTES in this file to ~10% above the "
        "new total. If it crept in via a docstring that grew long prose "
        "(band tables, bug narratives, multi-example JSON blocks), move "
        "that content to a skill reference under "
        "livepilot/skills/livepilot-core/references/ (or the relevant "
        "domain skill) and leave a one-line pointer instead — see "
        "perception.md / atlas-tool-notes.md / device-parameter-units.md "
        "for the established pattern."
    )


def test_no_single_tool_description_is_egregiously_oversized():
    """Catch one runaway tool description before it drags the whole budget.

    Not a hard content policy — just a smoke check that no single tool's
    description has silently grown into multi-KB territory (a red flag for
    an accidentally-pasted example block or an un-trimmed bug narrative).
    """
    from mcp_server.server import _get_all_tools

    tools = _get_all_tools()
    oversized = [
        (t.name, len(getattr(t, "description", "") or ""))
        for t in tools
        if len(getattr(t, "description", "") or "") > 4000
    ]
    assert not oversized, (
        f"{len(oversized)} tool description(s) exceed 4000 chars: "
        f"{oversized}. Move historical/narrative bulk to a skill reference "
        "and leave a one-line pointer in the docstring."
    )

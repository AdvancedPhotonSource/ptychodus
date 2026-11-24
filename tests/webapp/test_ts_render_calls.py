"""The chat panel draws every render tool the MCP server offers.

`ui/src/chat/render_calls.ts` maps each `render_*` tool to the REST route rendering the
same image, so the panel can draw a call itself. A tool missing from it would be called
by the agent and never shown.
"""

from __future__ import annotations

import re
from pathlib import Path

from fastmcp import FastMCP

_RENDER_CALLS_TS = (
    Path(__file__).parents[2]
    / 'src'
    / 'ptychodus'
    / 'webapp'
    / 'ui'
    / 'src'
    / 'chat'
    / 'render_calls.ts'
)


def _drawn_tools() -> set[str]:
    source = _RENDER_CALLS_TS.read_text()
    body = source[source.index('export const RENDER_CALLS') :]
    return set(re.findall(r'^\s{2}(render_\w+):', body, re.M))


async def test_every_render_tool_is_drawn_by_the_chat_panel(bound_mcp: FastMCP) -> None:
    tools = {tool.name for tool in await bound_mcp.list_tools() if tool.name.startswith('render_')}

    assert tools
    assert _drawn_tools() == tools

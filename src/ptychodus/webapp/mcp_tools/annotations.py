"""The tool annotations the webapp's MCP servers share."""

from __future__ import annotations

from mcp.types import ToolAnnotations

__all__ = ['READ_ONLY_ANNOTATIONS']

READ_ONLY_ANNOTATIONS = ToolAnnotations(
    read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=False
)
"""A tool that reads, changes nothing, gives the same answer when repeated, and reaches no
outside world."""

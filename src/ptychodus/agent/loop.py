"""The pydantic-ai agent: a language model with the webapp's MCP tools."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import partial

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.models import Model

from ptychodus.api.language_model import is_confidential_url

from .config import Settings
from .endpoint import create_model_settings
from .history import elide_earlier_images

logger = logging.getLogger(__name__)

__all__ = [
    'FORWARDED_USER_HEADER',
    'AgentDeps',
    'build_agent',
]

FORWARDED_USER_HEADER = 'X-Ptychodus-User'
"""The header naming, on each MCP call, the user a turn runs for."""


@dataclass(frozen=True)
class AgentDeps:
    """What one turn runs for."""

    user: str
    conversation_id: str


def _mcp_headers(token: str | None, ctx: RunContext[AgentDeps]) -> dict[str, str]:
    headers = {FORWARDED_USER_HEADER: ctx.deps.user}

    if token:
        headers['Authorization'] = f'Bearer {token}'

    return headers


def build_agent(language_model: Model, settings: Settings) -> Agent[AgentDeps, str]:
    """The agent the service runs every turn on.

    Its tools are the webapp's MCP server, reached with the service's token and the
    user of each turn, and the server's instructions join the agent's own.

    Raises:
        ValueError: if the token would cross the network in clear text.
    """
    token = settings.mcp_token.get_secret_value() if settings.mcp_token else None

    if token and not is_confidential_url(settings.mcp_url):
        raise ValueError(
            f'Refusing to send the MCP token to "{settings.mcp_url}"; use https:// or a '
            'loopback address.'
        )

    if not token:
        logger.warning('No PTYCHODUS_AGENT_MCP_TOKEN; the webapp refuses MCP without one.')

    toolset = MCPToolset[AgentDeps](
        settings.mcp_url,
        headers=partial(_mcp_headers, token),
        include_instructions=True,
    )
    return Agent(
        language_model,
        deps_type=AgentDeps,
        instructions=settings.instructions,
        model_settings=create_model_settings(settings.thinking, settings.max_tokens),
        toolsets=[toolset],
        capabilities=[ProcessHistory(partial(elide_earlier_images, keep_turns=1))],
    )

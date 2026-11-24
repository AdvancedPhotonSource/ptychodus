from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from typing import Any

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = [
    'Settings',
    'ThinkingLevel',
    'get_settings',
]


class ThinkingLevel(StrEnum):
    DEFAULT = 'Default'
    OFF = 'Off'
    LOW = 'Low'
    MEDIUM = 'Medium'
    HIGH = 'High'


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix='PTYCHODUS_AGENT_',
        env_file='.env',
        extra='ignore',
    )

    user_header: str = Field(..., min_length=1)
    """The request header the facility's reverse proxy sets to the signed-in user, e.g.
    ``X-Forwarded-User`` or ``Remote-User``. The service believes it, so only the proxy
    must be able to reach the service."""
    development_user: str | None = None
    """The user a request without `user_header` is taken to be. For development only,
    where no proxy stands in front of the service."""
    host: str = '127.0.0.1'
    port: int = 8001
    log_level: str = 'INFO'
    database_url: str = 'sqlite+aiosqlite:///:memory:'
    """Where conversations and the audit log are kept. In memory, both vanish on a
    restart."""

    provider: str = 'Argo'
    """The language-model provider plugin, by simple name."""
    model_id: str = ''
    """The language model to use; empty for the provider's default."""
    provider_options: dict[str, Any] = Field(default_factory=dict)
    """Fields of the provider's options, as JSON, over its defaults."""
    instructions: str = (
        'You are the assistant of ptychodus, a ptychography data-analysis application. '
        'Answer from what the tools report, and say when they do not settle a question. '
        'Write math in LaTeX between $...$ (inline) or $$...$$ (display), never \\(...\\) '
        'or \\[...\\].'
    )
    thinking: ThinkingLevel = ThinkingLevel.DEFAULT
    max_tokens: int = Field(0, ge=0)
    """The most tokens one response may hold; 0 leaves it to the language model."""

    mcp_url: str = 'http://127.0.0.1:8000/mcp/'
    """The webapp's MCP server."""
    mcp_token: SecretStr | None = None
    """The bearer token ``ptychodus-webapp mint-token`` printed for this service."""

    turn_request_limit: int | None = Field(20, ge=1)
    """The most language-model requests one turn may make."""
    turn_tool_calls_limit: int | None = Field(50, ge=1)
    """The most tool calls one turn may make."""
    turn_total_tokens_limit: int | None = Field(None, ge=1)
    """The most tokens one turn may spend."""
    user_daily_tokens_limit: int | None = Field(None, ge=1)
    """The most tokens one user may spend in any 24 hours, across conversations."""


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

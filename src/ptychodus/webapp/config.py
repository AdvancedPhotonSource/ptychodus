from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix='PTYCHODUS_WEBAPP_',
        env_file='.env',
        extra='ignore',
    )

    storage_root: Path = Field(...)
    database_url: str = 'sqlite+aiosqlite:///:memory:'
    polling_interval_s: float = 2.0
    debounce_window_s: float = 1.0
    log_level: str = 'INFO'
    host: str = '127.0.0.1'
    port: int = 8000
    mcp_mount_path: str = '/mcp'
    auto_reconcile_on_startup: bool = True
    render_concurrency: int = Field(4, ge=1)
    """How many bounded operations, renders and reports, may run at once, each in a worker
    thread holding its resources."""
    mcp_tokens_file: Path | None = None
    """The owner-only JSON file of MCP client token digests that ``ptychodus-webapp
    mint-token`` writes. MCP requires one unless `mcp_allow_unauthenticated` is set."""
    mcp_allow_unauthenticated: bool = False
    """Serve MCP to every caller when no token file is set. For development only: a
    reverse proxy in front of the webapp would expose MCP to everyone it reaches."""


class TokenFileSettings(BaseSettings):
    """The one setting the token commands read, so they need no storage root."""

    model_config = Settings.model_config

    mcp_tokens_file: Path | None = None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

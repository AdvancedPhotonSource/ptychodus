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
    api_prefix: str = '/api/v1'
    auto_reconcile_on_startup: bool = True
    render_concurrency: int = Field(4, ge=1)
    """How many bounded operations, renders and reports, may run at once, each in a worker
    thread holding its resources."""
    mcp_tokens_file: Path | None = None
    """The owner-only JSON file of MCP client token digests that ``ptychodus-webapp
    mint-token`` writes. Unset, MCP authenticates no one and is served only on loopback."""


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

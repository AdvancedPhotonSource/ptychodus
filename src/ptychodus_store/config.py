from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix='PTYCHODUS_STORE_',
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


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]

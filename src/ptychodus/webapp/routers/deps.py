"""FastAPI dependency providers."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from ptychodus.webapp.config import Settings
from ptychodus.webapp.db.session import SessionProvider
from ptychodus.api.storage import StorageLayout


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def get_layout(request: Request) -> StorageLayout:
    return request.app.state.layout


def get_session_provider(request: Request) -> SessionProvider:
    return request.app.state.session_provider


async def get_session(
    provider: Annotated[SessionProvider, Depends(get_session_provider)],
) -> AsyncIterator[AsyncSession]:
    async for session in provider.session():
        yield session


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
LayoutDep = Annotated[StorageLayout, Depends(get_layout)]
SessionDep = Annotated[AsyncSession, Depends(get_session)]

"""FastAPI dependency providers."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from ptychodus_store.config import Settings
from ptychodus_store.db.session import SessionProvider
from ptychodus_store.storage.layout import StoreLayout


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings  # type: ignore[no-any-return]


def get_layout(request: Request) -> StoreLayout:
    return request.app.state.layout  # type: ignore[no-any-return]


def get_session_provider(request: Request) -> SessionProvider:
    return request.app.state.session_provider  # type: ignore[no-any-return]


async def get_session(
    provider: Annotated[SessionProvider, Depends(get_session_provider)],
) -> AsyncIterator[AsyncSession]:
    async for session in provider.session():
        yield session


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
LayoutDep = Annotated[StoreLayout, Depends(get_layout)]
SessionDep = Annotated[AsyncSession, Depends(get_session)]

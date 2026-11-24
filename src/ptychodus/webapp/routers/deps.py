"""FastAPI dependency providers."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Annotated, Any

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from ptychodus.webapp.config import Settings
from ptychodus.webapp.db.session import SessionProvider
from ptychodus.webapp.service.operations import operation_class_of
from ptychodus.webapp.service.resources import ResourceService
from ptychodus.api.storage import StorageLayout


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def get_layout(request: Request) -> StorageLayout:
    return request.app.state.layout


def get_service(request: Request) -> ResourceService:
    return request.app.state.service


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
ServiceDep = Annotated[ResourceService, Depends(get_service)]


def openapi_operation(method: Callable[..., Any]) -> dict[str, Any]:
    """The OpenAPI extension stating the operation class a route's service method declares."""
    return {'x-operation-class': operation_class_of(method).value}

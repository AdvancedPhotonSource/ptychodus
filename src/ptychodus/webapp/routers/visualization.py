"""Options-enumeration endpoint for the visualization API."""

from __future__ import annotations

from fastapi import APIRouter

from ptychodus.webapp.rendering.schemas import OptionsRead
from ptychodus.webapp.routers.deps import ServiceDep, openapi_operation
from ptychodus.webapp.service.resources import ResourceService

router = APIRouter(prefix='/visualization', tags=['visualization'])


@router.get(
    '/options',
    response_model=OptionsRead,
    openapi_extra=openapi_operation(ResourceService.get_visualization_options),
)
async def get_visualization_options(service: ServiceDep) -> OptionsRead:
    """Enumerate the valid choices a client may pass to render endpoints."""
    return await service.get_visualization_options()

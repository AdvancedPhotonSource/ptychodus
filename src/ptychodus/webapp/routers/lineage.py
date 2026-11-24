from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter

from ptychodus.webapp.routers.deps import ServiceDep, openapi_operation
from ptychodus.webapp.service.resources import ResourceService
from ptychodus.webapp.service.schemas import LineageRead

router = APIRouter(tags=['lineage'])


@router.get(
    '/lineage/{uuid}',
    response_model=LineageRead,
    openapi_extra=openapi_operation(ResourceService.get_lineage),
)
async def get_lineage(uuid: UUID, service: ServiceDep) -> LineageRead:
    return await service.get_lineage(uuid)

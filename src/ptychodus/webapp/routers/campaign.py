from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Query

from ptychodus.webapp.db.base import IngestState
from ptychodus.webapp.routers.deps import ServiceDep, openapi_operation
from ptychodus.webapp.service.resources import ResourceService
from ptychodus.webapp.service.schemas import CampaignRead, Page

router = APIRouter(prefix='/campaign', tags=['campaign'])


@router.get(
    '',
    response_model=Page[CampaignRead],
    openapi_extra=openapi_operation(ResourceService.list_campaign),
)
async def list_campaigns(
    service: ServiceDep,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    sample_name: str | None = None,
    tag: str | None = None,
    ingest_state: IngestState | None = None,
) -> Page[CampaignRead]:
    return await service.list_campaign(
        limit=limit, offset=offset, sample_name=sample_name, tag=tag, ingest_state=ingest_state
    )


@router.get(
    '/{uuid}',
    response_model=CampaignRead,
    openapi_extra=openapi_operation(ResourceService.get_campaign),
)
async def get_campaign(uuid: UUID, service: ServiceDep) -> CampaignRead:
    return await service.get_campaign(uuid)

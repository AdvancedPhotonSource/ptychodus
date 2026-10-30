from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import String, func

from ptychodus.webapp.db import repositories as repo
from ptychodus.webapp.db.base import IngestState
from ptychodus.webapp.db.models import Campaign
from ptychodus.webapp.routers._convert import campaign_to_read
from ptychodus.webapp.routers.deps import SessionDep
from ptychodus.webapp.routers.schemas import CampaignRead, Page

router = APIRouter(prefix='/campaign', tags=['campaign'])


@router.get('', response_model=Page[CampaignRead])
async def list_campaigns(
    session: SessionDep,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    sample_name: str | None = None,
    tag: str | None = None,
    ingest_state: IngestState | None = None,
) -> Page[CampaignRead]:
    where = []
    if sample_name is not None:
        where.append(Campaign.sample_name == sample_name)
    if ingest_state is not None:
        where.append(Campaign.ingest_state == ingest_state)
    if tag is not None:
        # Match if the tag string appears as an element of the JSON array.
        # tags is stored as a JSON-encoded TEXT column, so we look for the
        # quoted JSON form of the value.
        where.append(func.instr(func.cast(Campaign.tags, String), f'"{tag}"') > 0)

    items, total = await repo.list_rows(session, Campaign, limit=limit, offset=offset, where=where)
    return Page(items=[campaign_to_read(i) for i in items], total=total, limit=limit, offset=offset)


@router.get('/{uuid}', response_model=CampaignRead)
async def get_campaign(uuid: UUID, session: SessionDep) -> CampaignRead:
    row = await repo.get_row(session, Campaign, uuid)
    if row is None:
        raise HTTPException(status_code=404, detail=f'campaign {uuid} not found')
    return campaign_to_read(row)

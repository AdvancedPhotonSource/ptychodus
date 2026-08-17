"""Helpers to convert ORM rows + edges into API read models."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from ptychodus_store.db import repositories as repo
from ptychodus_store.db.models import Diffraction, Fluorescence, Product
from ptychodus_store.routers.schemas import (
    CampaignRead,
    DerivedFromEdge,
    DiffractionRead,
    FluorescenceRead,
    ProductRead,
)


async def _edges_for(session: AsyncSession, uuid: UUID) -> list[DerivedFromEdge]:
    edges = await repo.outgoing_edges(session, uuid)
    return [DerivedFromEdge(kind=e.target_kind, uuid=e.target_uuid) for e in edges]  # type: ignore[arg-type]


async def diffraction_to_read(session: AsyncSession, row: Diffraction) -> DiffractionRead:
    data = DiffractionRead.model_validate(row)
    data.derived_from = await _edges_for(session, row.uuid)
    return data


async def product_to_read(session: AsyncSession, row: Product) -> ProductRead:
    data = ProductRead.model_validate(row)
    data.derived_from = await _edges_for(session, row.uuid)
    return data


async def fluorescence_to_read(session: AsyncSession, row: Fluorescence) -> FluorescenceRead:
    data = FluorescenceRead.model_validate(row)
    data.derived_from = await _edges_for(session, row.uuid)
    return data


def campaign_to_read(row: object) -> CampaignRead:
    return CampaignRead.model_validate(row)

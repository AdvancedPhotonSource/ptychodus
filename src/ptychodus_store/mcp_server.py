"""fastmcp server with read-only tools mirroring the REST surface."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID

from fastmcp import FastMCP
from sqlalchemy import String, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ptychodus_store.db import repositories as repo
from ptychodus_store.db.base import IngestState
from ptychodus_store.db.models import Campaign, DerivationEdge, Diffraction, Fluorescence, Product
from ptychodus_store.db.session import SessionProvider
from ptychodus_store.routers._convert import (
    campaign_to_read,
    diffraction_to_read,
    fluorescence_to_read,
    product_to_read,
)
from ptychodus_store.routers.schemas import (
    CampaignRead,
    DiffractionRead,
    FluorescenceRead,
    LineageNode,
    LineageRead,
    Page,
    ProductRead,
    StoreStats,
)
from ptychodus_store.storage.manifest import ResourceKind

logger = logging.getLogger(__name__)


class _Ctx:
    provider: SessionProvider | None = None


def bind_session_provider(provider: SessionProvider) -> None:
    _Ctx.provider = provider


@asynccontextmanager
async def _session() -> AsyncIterator[AsyncSession]:
    if _Ctx.provider is None:
        raise RuntimeError('SessionProvider not bound; call bind_session_provider() first')
    async with _Ctx.provider.session_factory() as session:
        yield session


def create_mcp_server() -> FastMCP:
    mcp = FastMCP(name='ptychodus-store')

    @mcp.tool()
    async def list_campaign(
        limit: int = 50,
        offset: int = 0,
        sample_name: str | None = None,
        ingest_state: str | None = None,
    ) -> Page[CampaignRead]:
        """List campaigns."""
        async with _session() as session:
            where = []
            if sample_name is not None:
                where.append(Campaign.sample_name == sample_name)
            if ingest_state is not None:
                where.append(Campaign.ingest_state == ingest_state)
            items, total = await repo.list_rows(
                session, ResourceKind.CAMPAIGN, limit=limit, offset=offset, where=where
            )
            return Page(
                items=[campaign_to_read(i) for i in items],
                total=total,
                limit=limit,
                offset=offset,
            )

    @mcp.tool()
    async def get_campaign(uuid: str) -> CampaignRead | None:
        """Get a campaign by UUID."""
        async with _session() as session:
            row = await repo.get_row(session, ResourceKind.CAMPAIGN, UUID(uuid))
            return campaign_to_read(row) if row is not None else None

    @mcp.tool()
    async def list_diffraction(
        limit: int = 50,
        offset: int = 0,
        campaign_uuid: str | None = None,
        derived_from_uuid: str | None = None,
        probe_energy_eV_min: float | None = None,  # noqa: N803
        probe_energy_eV_max: float | None = None,  # noqa: N803
        ingest_state: str | None = None,
    ) -> Page[DiffractionRead]:
        """List diffraction datasets."""
        async with _session() as session:
            where = []
            if campaign_uuid is not None:
                where.append(Diffraction.campaign_uuid == UUID(campaign_uuid))
            if ingest_state is not None:
                where.append(Diffraction.ingest_state == ingest_state)
            if probe_energy_eV_min is not None:
                where.append(Diffraction.probe_energy_eV >= probe_energy_eV_min)
            if probe_energy_eV_max is not None:
                where.append(Diffraction.probe_energy_eV <= probe_energy_eV_max)
            if derived_from_uuid is not None:
                target = UUID(derived_from_uuid)
                where.append(
                    Diffraction.uuid.in_(
                        select(DerivationEdge.source_uuid).where(
                            DerivationEdge.target_uuid == target
                        )
                    )
                )
            items, total = await repo.list_rows(
                session, ResourceKind.DIFFRACTION, limit=limit, offset=offset, where=where
            )
            reads = [await diffraction_to_read(session, i) for i in items]  # type: ignore[arg-type]
            return Page(items=reads, total=total, limit=limit, offset=offset)

    @mcp.tool()
    async def get_diffraction(uuid: str) -> DiffractionRead | None:
        """Get a diffraction dataset by UUID."""
        async with _session() as session:
            row = await repo.get_row(session, ResourceKind.DIFFRACTION, UUID(uuid))
            return await diffraction_to_read(session, row) if row is not None else None  # type: ignore[arg-type]

    @mcp.tool()
    async def list_product(
        limit: int = 50,
        offset: int = 0,
        derived_from_uuid: str | None = None,
        ingest_state: str | None = None,
    ) -> Page[ProductRead]:
        """List reconstruction products."""
        async with _session() as session:
            where = []
            if ingest_state is not None:
                where.append(Product.ingest_state == ingest_state)
            if derived_from_uuid is not None:
                target = UUID(derived_from_uuid)
                where.append(
                    Product.uuid.in_(
                        select(DerivationEdge.source_uuid).where(
                            DerivationEdge.target_uuid == target
                        )
                    )
                )
            items, total = await repo.list_rows(
                session, ResourceKind.PRODUCT, limit=limit, offset=offset, where=where
            )
            reads = [await product_to_read(session, i) for i in items]  # type: ignore[arg-type]
            return Page(items=reads, total=total, limit=limit, offset=offset)

    @mcp.tool()
    async def get_product(uuid: str) -> ProductRead | None:
        """Get a product by UUID."""
        async with _session() as session:
            row = await repo.get_row(session, ResourceKind.PRODUCT, UUID(uuid))
            return await product_to_read(session, row) if row is not None else None  # type: ignore[arg-type]

    @mcp.tool()
    async def list_fluorescence(
        limit: int = 50,
        offset: int = 0,
        derived_from_uuid: str | None = None,
        element: str | None = None,
    ) -> Page[FluorescenceRead]:
        """List fluorescence datasets."""
        async with _session() as session:
            where = []
            if derived_from_uuid is not None:
                target = UUID(derived_from_uuid)
                where.append(
                    Fluorescence.uuid.in_(
                        select(DerivationEdge.source_uuid).where(
                            DerivationEdge.target_uuid == target
                        )
                    )
                )
            if element is not None:
                where.append(
                    func.instr(func.cast(Fluorescence.element_names, String), f'"{element}"') > 0  # type: ignore[arg-type]
                )
            items, total = await repo.list_rows(
                session, ResourceKind.FLUORESCENCE, limit=limit, offset=offset, where=where
            )
            reads = [await fluorescence_to_read(session, i) for i in items]  # type: ignore[arg-type]
            return Page(items=reads, total=total, limit=limit, offset=offset)

    @mcp.tool()
    async def get_fluorescence(uuid: str) -> FluorescenceRead | None:
        """Get a fluorescence dataset by UUID."""
        async with _session() as session:
            row = await repo.get_row(session, ResourceKind.FLUORESCENCE, UUID(uuid))
            return await fluorescence_to_read(session, row) if row is not None else None  # type: ignore[arg-type]

    @mcp.tool()
    async def get_lineage(uuid: str) -> LineageRead | None:
        """Walk the derivation DAG up and down from a node."""
        from ptychodus_store.routers.lineage import (
            _find_campaign,
            _label_for,
            _resolve_node,
            _walk_ancestors,
            _walk_descendants,
        )

        target = UUID(uuid)
        async with _session() as session:
            resolved = await _resolve_node(session, target)
            if resolved is None:
                return None
            kind, row = resolved
            node = LineageNode(kind=kind, uuid=target, label=_label_for(row))  # type: ignore[arg-type]
            ancestors = await _walk_ancestors(session, target)
            descendants = await _walk_descendants(session, target)
            campaign = await _find_campaign(session, target, ancestors)
            return LineageRead(
                node=node, ancestors=ancestors, descendants=descendants, campaign=campaign
            )

    @mcp.tool()
    async def get_store_stats() -> StoreStats:
        """Return resource counts and invalid-row count for the store."""
        async with _session() as session:

            async def _count(model):  # type: ignore[no-untyped-def]
                return int(
                    (await session.execute(select(func.count()).select_from(model))).scalar_one()
                )

            invalid_total = 0
            for model in (Campaign, Diffraction, Product, Fluorescence):
                stmt = (
                    select(func.count())
                    .select_from(model)
                    .where(
                        model.ingest_state.in_(
                            [IngestState.INVALID, IngestState.MISSING_FILES, IngestState.ORPHANED]
                        )
                    )
                )
                invalid_total += int((await session.execute(stmt)).scalar_one())

            return StoreStats(
                campaign_count=await _count(Campaign),
                diffraction_count=await _count(Diffraction),
                product_count=await _count(Product),
                fluorescence_count=await _count(Fluorescence),
                invalid_count=invalid_total,
            )

    return mcp

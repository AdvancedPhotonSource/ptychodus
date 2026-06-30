from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from ptychodus_store.db.base import IngestState
from ptychodus_store.db.models import (
    Campaign,
    DerivationEdge,
    Diffraction,
    Fluorescence,
    Product,
)
from ptychodus_store.storage.manifest import ResourceKind

KIND_TO_MODEL: dict[str, type[Campaign | Diffraction | Product | Fluorescence]] = {
    ResourceKind.CAMPAIGN: Campaign,
    ResourceKind.DIFFRACTION: Diffraction,
    ResourceKind.PRODUCT: Product,
    ResourceKind.FLUORESCENCE: Fluorescence,
}


async def upsert_row(session: AsyncSession, kind: str, values: dict[str, Any]) -> None:
    """Insert or update one row keyed by `values['uuid']` on the table for `kind`."""
    model = KIND_TO_MODEL[kind]
    stmt = sqlite_insert(model).values(**values)
    update_cols = {k: stmt.excluded[k] for k in values if k != 'uuid'}
    stmt = stmt.on_conflict_do_update(index_elements=['uuid'], set_=update_cols)
    await session.execute(stmt)


async def delete_row(session: AsyncSession, kind: str, uuid: UUID) -> None:
    model = KIND_TO_MODEL[kind]
    await session.execute(delete(model).where(model.uuid == uuid))


async def get_row(
    session: AsyncSession, kind: str, uuid: UUID
) -> Campaign | Diffraction | Product | Fluorescence | None:
    model = KIND_TO_MODEL[kind]
    return await session.get(model, uuid)


async def list_rows(
    session: AsyncSession,
    kind: str,
    *,
    limit: int,
    offset: int,
    where: Sequence[Any] = (),
) -> tuple[list[Campaign | Diffraction | Product | Fluorescence], int]:
    model = KIND_TO_MODEL[kind]
    stmt = select(model)
    for clause in where:
        stmt = stmt.where(clause)
    stmt = stmt.order_by(model.created_at.desc()).limit(limit).offset(offset)
    raw_items = (await session.execute(stmt)).scalars().all()

    count_stmt = select(func.count()).select_from(model)
    for clause in where:
        count_stmt = count_stmt.where(clause)
    total = (await session.execute(count_stmt)).scalar_one()
    items: list[Campaign | Diffraction | Product | Fluorescence] = list(raw_items)  # type: ignore[arg-type]
    return items, int(total)


async def replace_edges(
    session: AsyncSession,
    source_kind: str,
    source_uuid: UUID,
    edges: Sequence[tuple[str, UUID]],
) -> None:
    """Clear all outgoing edges for `source_uuid` and reinsert from `edges`."""
    await session.execute(delete(DerivationEdge).where(DerivationEdge.source_uuid == source_uuid))
    if not edges:
        return
    rows = [
        {
            'source_uuid': source_uuid,
            'target_uuid': target_uuid,
            'source_kind': source_kind,
            'target_kind': target_kind,
        }
        for (target_kind, target_uuid) in edges
    ]
    await session.execute(sqlite_insert(DerivationEdge).values(rows))


async def edges_referencing(session: AsyncSession, target_uuid: UUID) -> list[DerivationEdge]:
    stmt = select(DerivationEdge).where(DerivationEdge.target_uuid == target_uuid)
    return list((await session.execute(stmt)).scalars().all())


async def outgoing_edges(session: AsyncSession, source_uuid: UUID) -> list[DerivationEdge]:
    stmt = select(DerivationEdge).where(DerivationEdge.source_uuid == source_uuid)
    return list((await session.execute(stmt)).scalars().all())


async def row_exists(session: AsyncSession, kind: str, uuid: UUID) -> bool:
    return (await get_row(session, kind, uuid)) is not None


async def update_state(session: AsyncSession, kind: str, uuid: UUID, state: IngestState) -> None:
    row = await get_row(session, kind, uuid)
    if row is not None:
        row.ingest_state = state


async def find_kind_for_uuid(session: AsyncSession, uuid: UUID) -> str | None:
    """Look up which table holds a row with this uuid (None if absent)."""
    for kind, model in KIND_TO_MODEL.items():
        if await session.get(model, uuid) is not None:
            return kind
    return None

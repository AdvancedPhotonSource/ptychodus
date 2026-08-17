from __future__ import annotations

from collections import deque
from uuid import UUID

from fastapi import APIRouter, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ptychodus_store.db import repositories as repo
from ptychodus_store.db.models import Campaign, DerivationEdge, Diffraction
from ptychodus_store.routers._convert import campaign_to_read
from ptychodus_store.routers.deps import SessionDep
from ptychodus_store.routers.schemas import LineageNode, LineageRead

router = APIRouter(tags=['lineage'])


def _label_for(row: object) -> str:
    return str(getattr(row, 'label', '') or getattr(row, 'name', '') or '')


async def _resolve_node(session: AsyncSession, uuid: UUID) -> tuple[str, object] | None:
    kind = await repo.find_kind_for_uuid(session, uuid)
    if kind is None:
        return None
    row = await repo.get_row(session, kind, uuid)
    if row is None:
        return None
    return kind, row


async def _walk_ancestors(session: AsyncSession, root: UUID) -> list[LineageNode]:
    """BFS over outgoing edges (child -> parent), cycle-guarded by visited set."""
    out: list[LineageNode] = []
    visited: set[UUID] = {root}
    queue: deque[UUID] = deque([root])
    while queue:
        current = queue.popleft()
        edges = await repo.outgoing_edges(session, current)
        for edge in edges:
            if edge.target_uuid in visited:
                continue
            visited.add(edge.target_uuid)
            resolved = await _resolve_node(session, edge.target_uuid)
            if resolved is None:
                continue
            kind, row = resolved
            out.append(LineageNode(kind=kind, uuid=edge.target_uuid, label=_label_for(row)))  # type: ignore[arg-type]
            queue.append(edge.target_uuid)
    return out


async def _walk_descendants(session: AsyncSession, root: UUID) -> list[LineageNode]:
    """BFS over incoming edges (parent -> child), cycle-guarded."""
    out: list[LineageNode] = []
    visited: set[UUID] = {root}
    queue: deque[UUID] = deque([root])
    while queue:
        current = queue.popleft()
        rows = (
            (
                await session.execute(
                    select(DerivationEdge).where(DerivationEdge.target_uuid == current)
                )
            )
            .scalars()
            .all()
        )
        for edge in rows:
            if edge.source_uuid in visited:
                continue
            visited.add(edge.source_uuid)
            resolved = await _resolve_node(session, edge.source_uuid)
            if resolved is None:
                continue
            kind, row = resolved
            out.append(LineageNode(kind=kind, uuid=edge.source_uuid, label=_label_for(row)))  # type: ignore[arg-type]
            queue.append(edge.source_uuid)
    return out


async def _find_campaign(session: AsyncSession, root_uuid: UUID, ancestors: list[LineageNode]):
    """Look at the root node and its ancestors for any diffraction → return its campaign."""
    candidates: list[UUID] = [root_uuid] + [a.uuid for a in ancestors]
    for uuid in candidates:
        diff = await session.get(Diffraction, uuid)
        if diff is not None and diff.campaign_uuid is not None:
            campaign = await session.get(Campaign, diff.campaign_uuid)
            if campaign is not None:
                return campaign_to_read(campaign)
    return None


@router.get('/lineage/{uuid}', response_model=LineageRead)
async def get_lineage(uuid: UUID, session: SessionDep) -> LineageRead:
    resolved = await _resolve_node(session, uuid)
    if resolved is None:
        raise HTTPException(status_code=404, detail=f'no resource with uuid {uuid}')
    kind, row = resolved
    node = LineageNode(kind=kind, uuid=uuid, label=_label_for(row))  # type: ignore[arg-type]

    ancestors = await _walk_ancestors(session, uuid)
    descendants = await _walk_descendants(session, uuid)
    campaign = await _find_campaign(session, uuid, ancestors)
    return LineageRead(node=node, ancestors=ancestors, descendants=descendants, campaign=campaign)

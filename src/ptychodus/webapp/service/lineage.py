"""Walking the derivation DAG the manifests' `derived_from` edges form."""

from __future__ import annotations

from collections import deque
from typing import cast
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ptychodus.webapp.db import repositories as repo
from ptychodus.webapp.db.models import Campaign, DerivationEdge, Diffraction
from ptychodus.webapp.service.convert import campaign_to_read, row_label
from ptychodus.webapp.service.schemas import CampaignRead, LineageNode, ResourceKindLiteral

__all__ = ['find_campaign', 'resolve_node', 'walk_ancestors', 'walk_descendants']


async def resolve_node(
    session: AsyncSession, uuid: UUID
) -> tuple[ResourceKindLiteral, object] | None:
    """The kind and row of the resource with `uuid`, or None if none is indexed."""
    kind = await repo.find_kind_for_uuid(session, uuid)
    if kind is None:
        return None
    row = await repo.get_row_by_kind(session, kind, uuid)
    if row is None:
        return None
    # find_kind_for_uuid only ever returns a KIND_TO_MODEL key, and those are exactly
    # the four literals LineageNode accepts.
    return cast(ResourceKindLiteral, kind), row


async def walk_ancestors(session: AsyncSession, root: UUID) -> list[LineageNode]:
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
            resolved = await resolve_node(session, edge.target_uuid)
            if resolved is None:
                continue
            kind, row = resolved
            out.append(LineageNode(kind=kind, uuid=edge.target_uuid, label=row_label(row)))
            queue.append(edge.target_uuid)
    return out


async def walk_descendants(session: AsyncSession, root: UUID) -> list[LineageNode]:
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
            resolved = await resolve_node(session, edge.source_uuid)
            if resolved is None:
                continue
            kind, row = resolved
            out.append(LineageNode(kind=kind, uuid=edge.source_uuid, label=row_label(row)))
            queue.append(edge.source_uuid)
    return out


async def find_campaign(
    session: AsyncSession, root_uuid: UUID, ancestors: list[LineageNode]
) -> CampaignRead | None:
    """Look at the root node and its ancestors for any diffraction → return its campaign."""
    candidates: list[UUID] = [root_uuid] + [a.uuid for a in ancestors]
    for uuid in candidates:
        diff = await session.get(Diffraction, uuid)
        if diff is not None and diff.campaign_uuid is not None:
            campaign = await session.get(Campaign, diff.campaign_uuid)
            if campaign is not None:
                return campaign_to_read(campaign)
    return None

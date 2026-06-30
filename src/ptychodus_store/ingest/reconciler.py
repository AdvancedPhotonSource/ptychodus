"""Full-store rescan: ingest every manifest, drop rows whose folder is gone."""

from __future__ import annotations

import logging
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from ptychodus_store.db import repositories as repo
from ptychodus_store.db.models import Campaign, DerivationEdge, Diffraction, Fluorescence, Product
from ptychodus_store.ingest.pipeline import ingest_manifest
from ptychodus_store.storage.layout import StoreLayout
from ptychodus_store.storage.manifest import ResourceKind

logger = logging.getLogger(__name__)


async def full_rescan(session: AsyncSession, layout: StoreLayout) -> dict[str, int]:
    """Walk the store, upsert every manifest, then drop rows whose folder no longer exists."""

    found_uuids: dict[str, set[UUID]] = {
        ResourceKind.CAMPAIGN: set(),
        ResourceKind.DIFFRACTION: set(),
        ResourceKind.PRODUCT: set(),
        ResourceKind.FLUORESCENCE: set(),
    }

    for manifest_path in layout.iter_manifest_paths():
        try:
            location = layout.parse_manifest_path(manifest_path)
        except Exception as exc:  # noqa: BLE001 — defensive
            logger.warning('skipping %s: %s', manifest_path, exc)
            continue
        await ingest_manifest(session, layout, manifest_path)
        found_uuids[location.kind].add(location.uuid)

    counts = {kind: len(uuids) for kind, uuids in found_uuids.items()}

    # Sweep DB-side rows whose folder vanished
    for kind, model in (
        (ResourceKind.CAMPAIGN, Campaign),
        (ResourceKind.DIFFRACTION, Diffraction),
        (ResourceKind.PRODUCT, Product),
        (ResourceKind.FLUORESCENCE, Fluorescence),
    ):
        existing_uuids = set((await session.execute(select(model.uuid))).scalars().all())
        stale = existing_uuids - found_uuids[kind]
        for uuid in stale:
            await repo.delete_row(session, kind, uuid)
            await session.execute(delete(DerivationEdge).where(DerivationEdge.source_uuid == uuid))

    await session.commit()
    return counts

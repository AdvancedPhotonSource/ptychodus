from __future__ import annotations

import asyncio
import logging
import uuid as uuid_lib
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, HTTPException

from ptychodus_store.ingest.pipeline import ingest_manifest
from ptychodus_store.ingest.reconciler import full_rescan
from ptychodus_store.routers.deps import LayoutDep, get_session_provider
from ptychodus_store.routers.schemas import ReindexResponse, StoreStats
from ptychodus_store.db.base import IngestState
from ptychodus_store.db.models import Campaign, Diffraction, Fluorescence, Product
from ptychodus_store.routers.deps import SessionDep
from ptychodus_store.storage.layout import LayoutError
from sqlalchemy import func, or_, select
from fastapi import Request

logger = logging.getLogger(__name__)
router = APIRouter(prefix='/admin', tags=['admin'])


# A full rescan walks the whole store and opens every HDF5 file it finds. The endpoint is
# unauthenticated, so without this guard each request piles another concurrent walk onto the
# same session factory.
_rescan_lock = asyncio.Lock()


async def _do_rescan(session_provider, layout) -> None:  # type: ignore[no-untyped-def]
    try:
        async with _rescan_lock:
            async with session_provider.session_factory() as session:
                await full_rescan(session, layout)
    except Exception:  # noqa: BLE001
        logger.exception('background reindex failed')


@router.post('/reindex', response_model=ReindexResponse)
async def reindex(
    request: Request, background_tasks: BackgroundTasks, layout: LayoutDep
) -> ReindexResponse:
    if _rescan_lock.locked():
        raise HTTPException(status_code=409, detail='a reindex is already in progress')

    job_id = str(uuid_lib.uuid4())
    session_provider = get_session_provider(request)
    background_tasks.add_task(asyncio.create_task, _do_rescan(session_provider, layout))
    return ReindexResponse(status='accepted', job_id=job_id)


@router.post('/rescan/{kind}/{uuid}')
async def rescan_one(
    kind: str, uuid: UUID, session: SessionDep, layout: LayoutDep
) -> dict[str, str]:
    try:
        folder = layout.resource_folder(kind, uuid)
    except LayoutError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    manifest_path = folder / 'manifest.json'
    if not manifest_path.is_file():
        raise HTTPException(status_code=404, detail=f'no manifest for {kind}/{uuid}')
    await ingest_manifest(session, layout, manifest_path)
    await session.commit()
    return {'status': 'rescanned', 'kind': kind, 'uuid': str(uuid)}


@router.get('/stats', response_model=StoreStats)
async def stats(session: SessionDep) -> StoreStats:
    async def _count(model) -> int:  # type: ignore[no-untyped-def]
        return int((await session.execute(select(func.count()).select_from(model))).scalar_one())

    async def _invalid_count() -> int:
        total = 0
        for model in (Campaign, Diffraction, Product, Fluorescence):
            stmt = (
                select(func.count())
                .select_from(model)
                .where(
                    or_(
                        model.ingest_state == IngestState.INVALID,
                        model.ingest_state == IngestState.MISSING_FILES,
                        model.ingest_state == IngestState.ORPHANED,
                    )
                )
            )
            total += int((await session.execute(stmt)).scalar_one())
        return total

    return StoreStats(
        campaign_count=await _count(Campaign),
        diffraction_count=await _count(Diffraction),
        product_count=await _count(Product),
        fluorescence_count=await _count(Fluorescence),
        invalid_count=await _invalid_count(),
    )

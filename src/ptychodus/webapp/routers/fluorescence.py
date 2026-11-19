from __future__ import annotations

from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy import String, exists, func, select


from ptychodus.api.processing import RunState

from ptychodus.webapp.db import repositories as repo
from ptychodus.webapp.db.base import IngestState
from ptychodus.webapp.db.models import DerivationEdge, Fluorescence, Product
from ptychodus.webapp.rendering import RenderedImage
from ptychodus.webapp.rendering.views import (
    fluorescence_element_view,
    fluorescence_summary_view,
    view_as_rendered_image,
)
from ptychodus.webapp.routers._render import render_or_http_error, resource_file_or_404
from ptychodus.webapp.rendering.params import RenderParamsDep
from ptychodus.webapp.routers._convert import fluorescence_to_read
from ptychodus.webapp.routers.deps import LayoutDep, RenderRunnerDep, SessionDep
from ptychodus.webapp.routers.schemas import FluorescenceRead, Page
from ptychodus.api.storage import ResourceKind

router = APIRouter(prefix='/fluorescence', tags=['fluorescence'])


@router.get('', response_model=Page[FluorescenceRead])
async def list_fluorescence(
    session: SessionDep,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    derived_from_uuid: UUID | None = None,
    element: str | None = None,
    ingest_state: IngestState | None = None,
    run_state: list[RunState] | None = Query(
        None, description='Only resources whose run is in one of these states; repeat for several.'
    ),
) -> Page[FluorescenceRead]:
    where = []
    if ingest_state is not None:
        where.append(Fluorescence.ingest_state == ingest_state)
    if run_state:
        where.append(Fluorescence.run_state.in_([s.value for s in run_state]))
    if element is not None:
        # JSON column contains the element string — use LIKE on the serialized JSON
        where.append(func.instr(func.cast(Fluorescence.element_names, String), f'"{element}"') > 0)
    if derived_from_uuid is not None:
        edge_subq = select(DerivationEdge.source_uuid).where(
            DerivationEdge.source_uuid == Fluorescence.uuid,
            DerivationEdge.target_uuid == derived_from_uuid,
        )
        where.append(exists(edge_subq))

    items, total = await repo.list_rows(
        session, Fluorescence, limit=limit, offset=offset, where=where
    )
    reads = [await fluorescence_to_read(session, i) for i in items]
    return Page(items=reads, total=total, limit=limit, offset=offset)


@router.get('/{uuid}', response_model=FluorescenceRead)
async def get_fluorescence(uuid: UUID, session: SessionDep) -> FluorescenceRead:
    row = await repo.get_row(session, Fluorescence, uuid)
    if row is None:
        raise HTTPException(status_code=404, detail=f'fluorescence {uuid} not found')
    return await fluorescence_to_read(session, row)


@router.get('/{uuid}/files/fluorescence')
async def get_fluorescence_file(uuid: UUID, session: SessionDep, layout: LayoutDep) -> FileResponse:
    path = await _fluorescence_file_or_404(uuid, session, layout)
    return FileResponse(path, media_type='application/x-hdf5', filename=path.name)


async def _fluorescence_file_or_404(uuid: UUID, session: SessionDep, layout: LayoutDep) -> Path:
    row = await repo.get_row(session, Fluorescence, uuid)
    if row is None:
        raise HTTPException(status_code=404, detail=f'fluorescence {uuid} not found')
    return resource_file_or_404(layout, ResourceKind.FLUORESCENCE, uuid)


async def _paired_product_file(
    product_uuid: UUID | None, session: SessionDep, layout: LayoutDep
) -> Path | None:
    """The paired product whose object pixel geometry a map takes, if one was named."""
    if product_uuid is None:
        return None
    row = await repo.get_row(session, Product, product_uuid)
    if row is None:
        raise HTTPException(
            status_code=404, detail=f'product {product_uuid} not found for pixel geometry'
        )
    return resource_file_or_404(layout, ResourceKind.PRODUCT, product_uuid)


@router.get('/{uuid}/image', response_model=RenderedImage)
async def get_fluorescence_summary_image(
    uuid: UUID,
    session: SessionDep,
    layout: LayoutDep,
    runner: RenderRunnerDep,
    params: RenderParamsDep,
    product_uuid: UUID | None = Query(
        None,
        description='Optional paired product for physical pixel geometry.',
    ),
) -> RenderedImage:
    """Render the elementwise sum of every element map in the dataset."""
    path = await _fluorescence_file_or_404(uuid, session, layout)
    product_path = await _paired_product_file(product_uuid, session, layout)
    return await render_or_http_error(
        runner,
        view_as_rendered_image,
        fluorescence_summary_view,
        (path, product_path),
        params,
        False,
    )


@router.get('/{uuid}/elements/{name}/image', response_model=RenderedImage)
async def get_fluorescence_element_image(
    uuid: UUID,
    name: str,
    session: SessionDep,
    layout: LayoutDep,
    runner: RenderRunnerDep,
    params: RenderParamsDep,
    product_uuid: UUID | None = Query(
        None,
        description='Optional paired product for physical pixel geometry.',
    ),
) -> RenderedImage:
    path = await _fluorescence_file_or_404(uuid, session, layout)
    product_path = await _paired_product_file(product_uuid, session, layout)
    return await render_or_http_error(
        runner,
        view_as_rendered_image,
        fluorescence_element_view,
        (path, name, product_path),
        params,
        False,
    )

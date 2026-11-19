from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy import exists, select

from ptychodus.api.scattering.geometry import ScatteringRegime

from ptychodus.api.processing import RunState

from ptychodus.webapp.db import repositories as repo
from ptychodus.webapp.db.base import IngestState
from ptychodus.webapp.db.models import DerivationEdge, Product
from ptychodus.webapp.rendering import RenderedImage
from ptychodus.webapp.rendering.schemas import PlotImage
from ptychodus.webapp.rendering.params import RenderParamsDep
from ptychodus.webapp.rendering.views import (
    object_layer_view,
    object_view,
    probe_modes_view,
    probe_view,
    scan_paths_view,
    view_as_rendered_image,
)
from ptychodus.webapp.routers._convert import product_to_read
from ptychodus.webapp.routers._render import render_or_http_error, resource_file_or_404
from ptychodus.webapp.routers.deps import LayoutDep, RenderRunnerDep, SessionDep
from ptychodus.webapp.routers.schemas import Page, ProductRead
from ptychodus.api.storage import ResourceKind

router = APIRouter(prefix='/product', tags=['product'])


@router.get('', response_model=Page[ProductRead])
async def list_product(
    session: SessionDep,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    derived_from_uuid: UUID | None = None,
    ingest_state: IngestState | None = None,
    run_state: list[RunState] | None = Query(
        None, description='Only resources whose run is in one of these states; repeat for several.'
    ),
    photon_energy_eV_min: float | None = Query(None, alias='photon_energy_eV_min'),  # noqa: N803
    photon_energy_eV_max: float | None = Query(None, alias='photon_energy_eV_max'),  # noqa: N803
    scattering_regime: ScatteringRegime | None = None,
) -> Page[ProductRead]:
    where = []
    if scattering_regime is not None:
        where.append(Product.scattering_regime == scattering_regime.value)
    if ingest_state is not None:
        where.append(Product.ingest_state == ingest_state)
    if run_state:
        where.append(Product.run_state.in_([s.value for s in run_state]))
    if photon_energy_eV_min is not None:
        where.append(Product.photon_energy_eV >= photon_energy_eV_min)
    if photon_energy_eV_max is not None:
        where.append(Product.photon_energy_eV <= photon_energy_eV_max)
    if derived_from_uuid is not None:
        edge_subq = select(DerivationEdge.source_uuid).where(
            DerivationEdge.source_uuid == Product.uuid,
            DerivationEdge.target_uuid == derived_from_uuid,
        )
        where.append(exists(edge_subq))

    items, total = await repo.list_rows(session, Product, limit=limit, offset=offset, where=where)
    reads = [await product_to_read(session, i) for i in items]
    return Page(items=reads, total=total, limit=limit, offset=offset)


@router.get('/{uuid}', response_model=ProductRead)
async def get_product(uuid: UUID, session: SessionDep) -> ProductRead:
    row = await repo.get_row(session, Product, uuid)
    if row is None:
        raise HTTPException(status_code=404, detail=f'product {uuid} not found')
    return await product_to_read(session, row)


@router.get('/{uuid}/files/product')
async def get_product_file(uuid: UUID, session: SessionDep, layout: LayoutDep) -> FileResponse:
    path = await _product_file_or_404(uuid, session, layout)
    return FileResponse(path, media_type='application/x-hdf5', filename=path.name)


async def _product_file_or_404(uuid: UUID, session: SessionDep, layout: LayoutDep) -> Path:
    """The product's current file: its final one, or the newest checkpoint while it runs."""
    row = await repo.get_row(session, Product, uuid)
    if row is None:
        raise HTTPException(status_code=404, detail=f'product {uuid} not found')
    return resource_file_or_404(layout, ResourceKind.PRODUCT, uuid)


@router.get('/{uuid}/probe/image', response_model=RenderedImage)
async def get_probe_image(
    uuid: UUID,
    session: SessionDep,
    layout: LayoutDep,
    runner: RenderRunnerDep,
    params: RenderParamsDep,
    incoherent: int = Query(0, ge=0, description='Incoherent probe mode index.'),
) -> RenderedImage:
    path = await _product_file_or_404(uuid, session, layout)
    return await render_or_http_error(
        runner, view_as_rendered_image, probe_view, (path, incoherent), params, True
    )


@router.get('/{uuid}/probe/modes/image', response_model=RenderedImage)
async def get_probe_modes_image(
    uuid: UUID,
    session: SessionDep,
    layout: LayoutDep,
    runner: RenderRunnerDep,
    params: RenderParamsDep,
) -> RenderedImage:
    path = await _product_file_or_404(uuid, session, layout)
    return await render_or_http_error(
        runner, view_as_rendered_image, probe_modes_view, (path,), params, True
    )


@router.get('/{uuid}/object/image', response_model=RenderedImage)
async def get_object_image(
    uuid: UUID,
    session: SessionDep,
    layout: LayoutDep,
    runner: RenderRunnerDep,
    params: RenderParamsDep,
) -> RenderedImage:
    """Render the object as the product of its layers, the transmission of the whole stack."""
    path = await _product_file_or_404(uuid, session, layout)
    return await render_or_http_error(
        runner, view_as_rendered_image, object_view, (path,), params, True
    )


@router.get('/{uuid}/object/{layer}/image', response_model=RenderedImage)
async def get_object_layer_image(
    uuid: UUID,
    layer: int,
    session: SessionDep,
    layout: LayoutDep,
    runner: RenderRunnerDep,
    params: RenderParamsDep,
) -> RenderedImage:
    path = await _product_file_or_404(uuid, session, layout)
    return await render_or_http_error(
        runner, view_as_rendered_image, object_layer_view, (path, layer), params, True
    )


@router.get('/positions/image', response_model=PlotImage)
async def get_positions_image(
    session: SessionDep,
    layout: LayoutDep,
    runner: RenderRunnerDep,
    uuid: Annotated[
        list[UUID],
        Query(description='Product to plot; repeat to overlay several on shared axes.'),
    ],
    connect_path: bool = Query(True, description='Join successive scan points with a line.'),
    width_px: int = Query(640, ge=128, le=2048),
    height_px: int = Query(640, ge=128, le=2048),
) -> PlotImage:
    """Plot one or more products' scan paths, one color and legend entry per product."""
    entries = [
        (await _product_file_or_404(product_uuid, session, layout), str(product_uuid)[:8])
        for product_uuid in uuid
    ]
    return await render_or_http_error(
        runner,
        partial(scan_paths_view, connect_path=connect_path, width_px=width_px, height_px=height_px),
        entries,
    )

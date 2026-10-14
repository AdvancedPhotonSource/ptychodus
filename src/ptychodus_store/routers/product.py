from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy import exists, select

from ptychodus.api.io import load_product
from ptychodus.api.product import Product as ProductAggregate

from ptychodus_store.db import repositories as repo
from ptychodus_store.db.base import IngestState
from ptychodus_store.db.models import DerivationEdge, Product
from ptychodus_store.rendering import RenderedImage, render_complex
from ptychodus_store.rendering.plot import ScanPath, render_scan_paths
from ptychodus_store.rendering.schemas import PlotImage
from ptychodus_store.rendering.params import RenderParamsDep
from ptychodus_store.routers._convert import product_to_read
from ptychodus_store.routers.deps import LayoutDep, SessionDep
from ptychodus_store.routers.schemas import Page, ProductRead
from ptychodus_store.storage.manifest import ResourceKind

router = APIRouter(prefix='/product', tags=['product'])


@router.get('', response_model=Page[ProductRead])
async def list_product(
    session: SessionDep,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    derived_from_uuid: UUID | None = None,
    ingest_state: IngestState | None = None,
    photon_energy_eV_min: float | None = Query(None, alias='photon_energy_eV_min'),  # noqa: N803
    photon_energy_eV_max: float | None = Query(None, alias='photon_energy_eV_max'),  # noqa: N803
) -> Page[ProductRead]:
    where = []
    if ingest_state is not None:
        where.append(Product.ingest_state == ingest_state)
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
    row = await repo.get_row(session, Product, uuid)
    if row is None:
        raise HTTPException(status_code=404, detail=f'product {uuid} not found')
    path = layout.resource_folder(ResourceKind.PRODUCT, uuid) / 'product.h5'
    if not path.is_file():
        raise HTTPException(status_code=404, detail='product.h5 not present on disk')
    return FileResponse(path, media_type='application/x-hdf5', filename=path.name)


async def _load_product_or_404(
    uuid: UUID, session: SessionDep, layout: LayoutDep
) -> ProductAggregate:
    row = await repo.get_row(session, Product, uuid)
    if row is None:
        raise HTTPException(status_code=404, detail=f'product {uuid} not found')
    path = layout.resource_folder(ResourceKind.PRODUCT, uuid) / 'product.h5'
    if not path.is_file():
        raise HTTPException(status_code=404, detail='product.h5 not present on disk')
    return load_product(path)


@router.get('/{uuid}/probe/image', response_model=RenderedImage)
async def get_probe_image(
    uuid: UUID,
    session: SessionDep,
    layout: LayoutDep,
    params: RenderParamsDep,
    incoherent: int = Query(0, ge=0, description='Incoherent probe mode index.'),
) -> RenderedImage:
    product = await _load_product_or_404(uuid, session, layout)
    probe = product.probes.get_probe_no_opr()
    if not 0 <= incoherent < probe.num_incoherent_modes:
        raise HTTPException(
            status_code=404,
            detail=(f'incoherent mode {incoherent} out of range [0, {probe.num_incoherent_modes})'),
        )
    values = probe.get_incoherent_mode(incoherent)
    return render_complex(values, product.probes.get_pixel_geometry(), params)


@router.get('/{uuid}/probe/modes/image', response_model=RenderedImage)
async def get_probe_modes_image(
    uuid: UUID,
    session: SessionDep,
    layout: LayoutDep,
    params: RenderParamsDep,
) -> RenderedImage:
    product = await _load_product_or_404(uuid, session, layout)
    probe = product.probes.get_probe_no_opr()
    values = probe.get_incoherent_modes_flattened()
    return render_complex(values, product.probes.get_pixel_geometry(), params)


@router.get('/{uuid}/object/{layer}/image', response_model=RenderedImage)
async def get_object_layer_image(
    uuid: UUID,
    layer: int,
    session: SessionDep,
    layout: LayoutDep,
    params: RenderParamsDep,
) -> RenderedImage:
    product = await _load_product_or_404(uuid, session, layout)
    if not 0 <= layer < product.object_.num_layers:
        raise HTTPException(
            status_code=404,
            detail=f'object layer {layer} out of range [0, {product.object_.num_layers})',
        )
    values = product.object_.get_layer(layer)
    return render_complex(values, product.object_.get_pixel_geometry(), params)


@router.get('/positions/image', response_model=PlotImage)
async def get_positions_image(
    session: SessionDep,
    layout: LayoutDep,
    uuid: Annotated[
        list[UUID],
        Query(description='Product to plot; repeat to overlay several on shared axes.'),
    ],
    connect_path: bool = Query(True, description='Join successive scan points with a line.'),
    width_px: int = Query(640, ge=128, le=2048),
    height_px: int = Query(640, ge=128, le=2048),
) -> PlotImage:
    """Plot one or more products' scan paths, one color and legend entry per product."""
    scans: list[ScanPath] = []

    for product_uuid in uuid:
        product = await _load_product_or_404(product_uuid, session, layout)
        positions = product.probe_positions

        if len(positions) == 0:
            raise HTTPException(
                status_code=404, detail=f'product {product_uuid} has no probe positions'
            )

        scans.append(
            ScanPath(
                label=product.metadata.name or str(product_uuid)[:8],
                x_m=[p.x_m for p in positions],
                y_m=[p.y_m for p in positions],
            )
        )

    return render_scan_paths(
        scans, connect_path=connect_path, width_px=width_px, height_px=height_px
    )

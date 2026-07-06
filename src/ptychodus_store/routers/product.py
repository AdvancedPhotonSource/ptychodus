from __future__ import annotations

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
    probe_energy_eV_min: float | None = Query(None, alias='probe_energy_eV_min'),  # noqa: N803
    probe_energy_eV_max: float | None = Query(None, alias='probe_energy_eV_max'),  # noqa: N803
) -> Page[ProductRead]:
    where = []
    if ingest_state is not None:
        where.append(Product.ingest_state == ingest_state)
    if probe_energy_eV_min is not None:
        where.append(Product.probe_energy_eV >= probe_energy_eV_min)
    if probe_energy_eV_max is not None:
        where.append(Product.probe_energy_eV <= probe_energy_eV_max)
    if derived_from_uuid is not None:
        edge_subq = select(DerivationEdge.source_uuid).where(
            DerivationEdge.source_uuid == Product.uuid,
            DerivationEdge.target_uuid == derived_from_uuid,
        )
        where.append(exists(edge_subq))

    items, total = await repo.list_rows(
        session, ResourceKind.PRODUCT, limit=limit, offset=offset, where=where
    )
    reads = [await product_to_read(session, i) for i in items]  # type: ignore[arg-type]
    return Page(items=reads, total=total, limit=limit, offset=offset)


@router.get('/{uuid}', response_model=ProductRead)
async def get_product(uuid: UUID, session: SessionDep) -> ProductRead:
    row = await repo.get_row(session, ResourceKind.PRODUCT, uuid)
    if row is None:
        raise HTTPException(status_code=404, detail=f'product {uuid} not found')
    return await product_to_read(session, row)  # type: ignore[arg-type]


@router.get('/{uuid}/files/product')
async def get_product_file(uuid: UUID, session: SessionDep, layout: LayoutDep) -> FileResponse:
    row = await repo.get_row(session, ResourceKind.PRODUCT, uuid)
    if row is None:
        raise HTTPException(status_code=404, detail=f'product {uuid} not found')
    path = layout.resource_folder(ResourceKind.PRODUCT, uuid) / 'product.h5'
    if not path.is_file():
        raise HTTPException(status_code=404, detail='product.h5 not present on disk')
    return FileResponse(path, media_type='application/x-hdf5', filename=path.name)


async def _load_product_or_404(
    uuid: UUID, session: SessionDep, layout: LayoutDep
) -> ProductAggregate:
    row = await repo.get_row(session, ResourceKind.PRODUCT, uuid)
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

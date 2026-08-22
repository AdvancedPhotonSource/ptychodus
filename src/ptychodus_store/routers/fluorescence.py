from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy import String, exists, func, select

from ptychodus.api.constants import LengthUnit
from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.io import load_fluorescence_data, load_product

from ptychodus_store.db import repositories as repo
from ptychodus_store.db.base import IngestState
from ptychodus_store.db.models import DerivationEdge, Fluorescence
from ptychodus_store.rendering import RenderedImage, render_real
from ptychodus_store.rendering.params import RenderParamsDep
from ptychodus_store.routers._convert import fluorescence_to_read
from ptychodus_store.routers.deps import LayoutDep, SessionDep
from ptychodus_store.routers.schemas import FluorescenceRead, Page
from ptychodus_store.storage.manifest import ResourceKind

_DEFAULT_FLUORESCENCE_PIXEL_GEOMETRY = PixelGeometry(
    width_m=LengthUnit.MICROMETER.meters_per_unit,
    height_m=LengthUnit.MICROMETER.meters_per_unit,
)

router = APIRouter(prefix='/fluorescence', tags=['fluorescence'])


@router.get('', response_model=Page[FluorescenceRead])
async def list_fluorescence(
    session: SessionDep,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    derived_from_uuid: UUID | None = None,
    element: str | None = None,
    ingest_state: IngestState | None = None,
) -> Page[FluorescenceRead]:
    where = []
    if ingest_state is not None:
        where.append(Fluorescence.ingest_state == ingest_state)
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
        session, ResourceKind.FLUORESCENCE, limit=limit, offset=offset, where=where
    )
    reads = [await fluorescence_to_read(session, i) for i in items]  # type: ignore[arg-type]
    return Page(items=reads, total=total, limit=limit, offset=offset)


@router.get('/{uuid}', response_model=FluorescenceRead)
async def get_fluorescence(uuid: UUID, session: SessionDep) -> FluorescenceRead:
    row = await repo.get_row(session, ResourceKind.FLUORESCENCE, uuid)
    if row is None:
        raise HTTPException(status_code=404, detail=f'fluorescence {uuid} not found')
    return await fluorescence_to_read(session, row)  # type: ignore[arg-type]


@router.get('/{uuid}/files/fluorescence')
async def get_fluorescence_file(uuid: UUID, session: SessionDep, layout: LayoutDep) -> FileResponse:
    row = await repo.get_row(session, ResourceKind.FLUORESCENCE, uuid)
    if row is None:
        raise HTTPException(status_code=404, detail=f'fluorescence {uuid} not found')
    path = layout.resource_folder(ResourceKind.FLUORESCENCE, uuid) / 'fluorescence.h5'
    if not path.is_file():
        raise HTTPException(status_code=404, detail='fluorescence.h5 not present on disk')
    return FileResponse(path, media_type='application/x-hdf5', filename=path.name)


async def _resolve_product_pixel_geometry(
    product_uuid: UUID | None, session: SessionDep, layout: LayoutDep
) -> PixelGeometry:
    """Look up a paired product's object pixel geometry; fall back to a 1 µm placeholder."""
    if product_uuid is None:
        return _DEFAULT_FLUORESCENCE_PIXEL_GEOMETRY
    row = await repo.get_row(session, ResourceKind.PRODUCT, product_uuid)
    if row is None:
        raise HTTPException(
            status_code=404, detail=f'product {product_uuid} not found for pixel geometry'
        )
    path = layout.resource_folder(ResourceKind.PRODUCT, product_uuid) / 'product.h5'
    if not path.is_file():
        raise HTTPException(status_code=404, detail='product.h5 not present on disk')
    return load_product(path).object_.get_pixel_geometry()


@router.get('/{uuid}/elements/{name}/image', response_model=RenderedImage)
async def get_fluorescence_element_image(
    uuid: UUID,
    name: str,
    session: SessionDep,
    layout: LayoutDep,
    params: RenderParamsDep,
    product_uuid: UUID | None = Query(
        None,
        description='Optional paired product for physical pixel geometry.',
    ),
) -> RenderedImage:
    row = await repo.get_row(session, ResourceKind.FLUORESCENCE, uuid)
    if row is None:
        raise HTTPException(status_code=404, detail=f'fluorescence {uuid} not found')
    path = layout.resource_folder(ResourceKind.FLUORESCENCE, uuid) / 'fluorescence.h5'
    if not path.is_file():
        raise HTTPException(status_code=404, detail='fluorescence.h5 not present on disk')

    dataset = load_fluorescence_data(path)
    matches = [emap for emap in dataset.element_maps if emap.name == name]
    if not matches:
        raise HTTPException(
            status_code=404,
            detail={
                'error': f'element {name!r} not found',
                'available': [emap.name for emap in dataset.element_maps],
            },
        )
    emap = matches[0]

    pixel_geometry = await _resolve_product_pixel_geometry(product_uuid, session, layout)
    return render_real(
        emap.counts_per_second, pixel_geometry, params, value_label=f'{emap.name} counts/s'
    )

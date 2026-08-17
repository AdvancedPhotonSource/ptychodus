from __future__ import annotations

from base64 import b64encode
from io import BytesIO
from uuid import UUID

import numpy
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from PIL import Image, ImageDraw
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


@router.get('/{uuid}/positions/image', response_model=RenderedImage)
async def get_positions_image(
    uuid: UUID,
    session: SessionDep,
    layout: LayoutDep,
    canvas_px: int = Query(512, ge=64, le=2048, description='Square output resolution in pixels.'),
    connect_path: bool = Query(True, description='Draw a polyline through successive scan points.'),
    margin_frac: float = Query(
        0.05, ge=0.0, le=0.5, description='Blank margin around the bounding box, as a fraction.'
    ),
) -> RenderedImage:
    product = await _load_product_or_404(uuid, session, layout)
    positions = product.probe_positions
    n_points = len(positions)
    if n_points == 0:
        raise HTTPException(status_code=404, detail='product has no probe positions')

    xs = numpy.array([positions[i].coordinate_x_m for i in range(n_points)], dtype=float)
    ys = numpy.array([positions[i].coordinate_y_m for i in range(n_points)], dtype=float)

    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    width_m = max(x_max - x_min, 1e-9)
    height_m = max(y_max - y_min, 1e-9)
    range_m = max(width_m, height_m) * (1.0 + 2.0 * margin_frac)
    x_center = 0.5 * (x_min + x_max)
    y_center = 0.5 * (y_min + y_max)

    scale = canvas_px / range_m
    px_x = (xs - x_center) * scale + canvas_px / 2.0
    px_y = canvas_px / 2.0 - (ys - y_center) * scale  # flip y so +y is up
    pts = [(int(round(px)), int(round(py))) for px, py in zip(px_x, px_y)]

    img = Image.new('RGB', (canvas_px, canvas_px), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    if connect_path and n_points > 1:
        draw.line(pts, fill=(80, 80, 80), width=1)
    radius = max(1, canvas_px // 200)
    denom = max(1, n_points - 1)
    for i, (px, py) in enumerate(pts):
        t = i / denom
        color = (int(255 * t), 80, int(255 * (1.0 - t)))
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=color)

    buf = BytesIO()
    img.save(buf, format='PNG')
    pixel_m = range_m / canvas_px
    return RenderedImage(
        png_base64=b64encode(buf.getvalue()).decode('ascii'),
        value_label='Scan Index',
        color_value_min=0.0,
        color_value_max=float(max(0, n_points - 1)),
        pixel_width_m=pixel_m,
        pixel_height_m=pixel_m,
        shape_h_px=canvas_px,
        shape_w_px=canvas_px,
    )

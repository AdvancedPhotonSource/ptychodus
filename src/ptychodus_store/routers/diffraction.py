from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy import exists, select

from ptychodus.api.diffraction import Polarization
from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.io import load_diffraction_data
from ptychodus.api.reconstructor import AssembledDiffractionData

from ptychodus_store.db import repositories as repo
from ptychodus_store.db.base import IngestState
from ptychodus_store.db.models import DerivationEdge, Diffraction
from ptychodus_store.rendering import RenderedImage, render_real
from ptychodus_store.rendering.params import RenderParamsDep
from ptychodus_store.routers._convert import diffraction_to_read
from ptychodus_store.routers.deps import LayoutDep, SessionDep
from ptychodus_store.routers.schemas import DiffractionRead, Page
from ptychodus_store.storage.manifest import ResourceKind

router = APIRouter(prefix='/diffraction', tags=['diffraction'])


@router.get('', response_model=Page[DiffractionRead])
async def list_diffraction(
    session: SessionDep,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    campaign_uuid: UUID | None = None,
    derived_from_uuid: UUID | None = None,
    ingest_state: IngestState | None = None,
    probe_energy_eV_min: float | None = Query(None, alias='probe_energy_eV_min'),  # noqa: N803
    probe_energy_eV_max: float | None = Query(None, alias='probe_energy_eV_max'),  # noqa: N803
    tomography_angle_deg_min: float | None = None,
    tomography_angle_deg_max: float | None = None,
    tilt_angle_deg_min: float | None = None,
    tilt_angle_deg_max: float | None = None,
    polarization: Polarization | None = None,
) -> Page[DiffractionRead]:
    where = []
    if campaign_uuid is not None:
        where.append(Diffraction.campaign_uuid == campaign_uuid)
    if ingest_state is not None:
        where.append(Diffraction.ingest_state == ingest_state)
    if probe_energy_eV_min is not None:
        where.append(Diffraction.probe_energy_eV >= probe_energy_eV_min)
    if probe_energy_eV_max is not None:
        where.append(Diffraction.probe_energy_eV <= probe_energy_eV_max)
    if tomography_angle_deg_min is not None:
        where.append(Diffraction.tomography_angle_deg >= tomography_angle_deg_min)
    if tomography_angle_deg_max is not None:
        where.append(Diffraction.tomography_angle_deg <= tomography_angle_deg_max)
    if tilt_angle_deg_min is not None:
        where.append(Diffraction.tilt_angle_deg >= tilt_angle_deg_min)
    if tilt_angle_deg_max is not None:
        where.append(Diffraction.tilt_angle_deg <= tilt_angle_deg_max)
    if polarization is not None:
        where.append(Diffraction.polarization == polarization.value)
    if derived_from_uuid is not None:
        edge_subq = select(DerivationEdge.source_uuid).where(
            DerivationEdge.source_uuid == Diffraction.uuid,
            DerivationEdge.target_uuid == derived_from_uuid,
        )
        where.append(exists(edge_subq))

    items, total = await repo.list_rows(
        session, ResourceKind.DIFFRACTION, limit=limit, offset=offset, where=where
    )
    reads = [await diffraction_to_read(session, i) for i in items]  # type: ignore[arg-type]
    return Page(items=reads, total=total, limit=limit, offset=offset)


@router.get('/{uuid}', response_model=DiffractionRead)
async def get_diffraction(uuid: UUID, session: SessionDep) -> DiffractionRead:
    row = await repo.get_row(session, ResourceKind.DIFFRACTION, uuid)
    if row is None:
        raise HTTPException(status_code=404, detail=f'diffraction {uuid} not found')
    return await diffraction_to_read(session, row)  # type: ignore[arg-type]


@router.get('/{uuid}/files/diffraction')
async def get_diffraction_file(uuid: UUID, session: SessionDep, layout: LayoutDep) -> FileResponse:
    row = await repo.get_row(session, ResourceKind.DIFFRACTION, uuid)
    if row is None:
        raise HTTPException(status_code=404, detail=f'diffraction {uuid} not found')
    path = layout.resource_folder(ResourceKind.DIFFRACTION, uuid) / 'diffraction.h5'
    if not path.is_file():
        raise HTTPException(status_code=404, detail='diffraction.h5 not present on disk')
    return FileResponse(path, media_type='application/x-hdf5', filename=path.name)


async def _load_diffraction_or_404(
    uuid: UUID, session: SessionDep, layout: LayoutDep
) -> tuple[AssembledDiffractionData, PixelGeometry]:
    row = await repo.get_row(session, ResourceKind.DIFFRACTION, uuid)
    if row is None:
        raise HTTPException(status_code=404, detail=f'diffraction {uuid} not found')
    path = layout.resource_folder(ResourceKind.DIFFRACTION, uuid) / 'diffraction.h5'
    if not path.is_file():
        raise HTTPException(status_code=404, detail='diffraction.h5 not present on disk')
    data = load_diffraction_data(path)
    return data, data.get_pixel_geometry()


@router.get('/{uuid}/patterns/aggregate/image', response_model=RenderedImage)
async def get_diffraction_aggregate_image(
    uuid: UUID,
    session: SessionDep,
    layout: LayoutDep,
    params: RenderParamsDep,
) -> RenderedImage:
    data, pixel_geometry = await _load_diffraction_or_404(uuid, session, layout)
    return render_real(
        data.get_average_pattern(), pixel_geometry, params, value_label='Mean Counts'
    )


@router.get('/{uuid}/patterns/{index}/image', response_model=RenderedImage)
async def get_diffraction_pattern_image(
    uuid: UUID,
    index: int,
    session: SessionDep,
    layout: LayoutDep,
    params: RenderParamsDep,
) -> RenderedImage:
    data, pixel_geometry = await _load_diffraction_or_404(uuid, session, layout)
    num_patterns = data.get_patterns_shape()[0]
    if not 0 <= index < num_patterns:
        raise HTTPException(
            status_code=404,
            detail=f'pattern index {index} out of range [0, {num_patterns})',
        )
    return render_real(data.get_pattern(index), pixel_geometry, params, value_label='Counts')

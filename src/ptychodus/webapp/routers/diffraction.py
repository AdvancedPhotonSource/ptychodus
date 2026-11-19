from __future__ import annotations

from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy import exists, select

from ptychodus.api.diffraction import Polarization

from ptychodus.api.processing import RunState

from ptychodus.webapp.db import repositories as repo
from ptychodus.webapp.db.base import IngestState
from ptychodus.webapp.db.models import DerivationEdge, Diffraction
from ptychodus.webapp.rendering import RenderedImage
from ptychodus.webapp.rendering.views import (
    diffraction_aggregate_view,
    diffraction_pattern_view,
    view_as_rendered_image,
)
from ptychodus.webapp.routers._render import render_or_http_error, resource_file_or_404
from ptychodus.webapp.rendering.params import RenderParamsDep
from ptychodus.webapp.routers._convert import diffraction_to_read
from ptychodus.webapp.routers.deps import LayoutDep, RenderRunnerDep, SessionDep
from ptychodus.webapp.routers.schemas import DiffractionRead, Page
from ptychodus.api.storage import ResourceKind

router = APIRouter(prefix='/diffraction', tags=['diffraction'])


@router.get('', response_model=Page[DiffractionRead])
async def list_diffraction(
    session: SessionDep,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    campaign_uuid: UUID | None = None,
    derived_from_uuid: UUID | None = None,
    ingest_state: IngestState | None = None,
    run_state: list[RunState] | None = Query(
        None, description='Only resources whose run is in one of these states; repeat for several.'
    ),
    photon_energy_eV_min: float | None = Query(None, alias='photon_energy_eV_min'),  # noqa: N803
    photon_energy_eV_max: float | None = Query(None, alias='photon_energy_eV_max'),  # noqa: N803
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
    if run_state:
        where.append(Diffraction.run_state.in_([s.value for s in run_state]))
    if photon_energy_eV_min is not None:
        where.append(Diffraction.photon_energy_eV >= photon_energy_eV_min)
    if photon_energy_eV_max is not None:
        where.append(Diffraction.photon_energy_eV <= photon_energy_eV_max)
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
        session, Diffraction, limit=limit, offset=offset, where=where
    )
    reads = [await diffraction_to_read(session, i) for i in items]
    return Page(items=reads, total=total, limit=limit, offset=offset)


@router.get('/{uuid}', response_model=DiffractionRead)
async def get_diffraction(uuid: UUID, session: SessionDep) -> DiffractionRead:
    row = await repo.get_row(session, Diffraction, uuid)
    if row is None:
        raise HTTPException(status_code=404, detail=f'diffraction {uuid} not found')
    return await diffraction_to_read(session, row)


@router.get('/{uuid}/files/diffraction')
async def get_diffraction_file(uuid: UUID, session: SessionDep, layout: LayoutDep) -> FileResponse:
    path = await _diffraction_file_or_404(uuid, session, layout)
    return FileResponse(path, media_type='application/x-hdf5', filename=path.name)


async def _diffraction_file_or_404(uuid: UUID, session: SessionDep, layout: LayoutDep) -> Path:
    row = await repo.get_row(session, Diffraction, uuid)
    if row is None:
        raise HTTPException(status_code=404, detail=f'diffraction {uuid} not found')
    return resource_file_or_404(layout, ResourceKind.DIFFRACTION, uuid)


@router.get('/{uuid}/patterns/aggregate/image', response_model=RenderedImage)
async def get_diffraction_aggregate_image(
    uuid: UUID,
    session: SessionDep,
    layout: LayoutDep,
    runner: RenderRunnerDep,
    params: RenderParamsDep,
) -> RenderedImage:
    path = await _diffraction_file_or_404(uuid, session, layout)
    return await render_or_http_error(
        runner, view_as_rendered_image, diffraction_aggregate_view, (path,), params, False
    )


@router.get('/{uuid}/patterns/{index}/image', response_model=RenderedImage)
async def get_diffraction_pattern_image(
    uuid: UUID,
    index: int,
    session: SessionDep,
    layout: LayoutDep,
    runner: RenderRunnerDep,
    params: RenderParamsDep,
) -> RenderedImage:
    path = await _diffraction_file_or_404(uuid, session, layout)
    return await render_or_http_error(
        runner, view_as_rendered_image, diffraction_pattern_view, (path, index), params, False
    )

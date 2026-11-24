from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Query
from fastapi.responses import FileResponse

from ptychodus.api.diffraction import Polarization
from ptychodus.api.processing import RunState
from ptychodus.api.storage import ResourceKind

from ptychodus.webapp.db.base import IngestState
from ptychodus.webapp.rendering import RenderedImage
from ptychodus.webapp.rendering.params import RenderParamsDep
from ptychodus.webapp.routers.deps import ServiceDep, openapi_operation
from ptychodus.webapp.service.resources import ResourceService
from ptychodus.webapp.service.schemas import DiffractionRead, Page

router = APIRouter(prefix='/diffraction', tags=['diffraction'])


@router.get(
    '',
    response_model=Page[DiffractionRead],
    openapi_extra=openapi_operation(ResourceService.list_diffraction),
)
async def list_diffraction(
    service: ServiceDep,
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
    return await service.list_diffraction(
        limit=limit,
        offset=offset,
        campaign_uuid=campaign_uuid,
        derived_from_uuid=derived_from_uuid,
        ingest_state=ingest_state,
        run_state=run_state,
        photon_energy_eV_min=photon_energy_eV_min,
        photon_energy_eV_max=photon_energy_eV_max,
        tomography_angle_deg_min=tomography_angle_deg_min,
        tomography_angle_deg_max=tomography_angle_deg_max,
        tilt_angle_deg_min=tilt_angle_deg_min,
        tilt_angle_deg_max=tilt_angle_deg_max,
        polarization=polarization,
    )


@router.get(
    '/{uuid}',
    response_model=DiffractionRead,
    openapi_extra=openapi_operation(ResourceService.get_diffraction),
)
async def get_diffraction(uuid: UUID, service: ServiceDep) -> DiffractionRead:
    return await service.get_diffraction(uuid)


@router.get('/{uuid}/files/diffraction')
async def get_diffraction_file(uuid: UUID, service: ServiceDep) -> FileResponse:
    path = await service.resource_file(ResourceKind.DIFFRACTION, uuid)
    return FileResponse(path, media_type='application/x-hdf5', filename=path.name)


@router.get(
    '/{uuid}/patterns/aggregate/image',
    response_model=RenderedImage,
    openapi_extra=openapi_operation(ResourceService.render_diffraction_aggregate),
)
async def get_diffraction_aggregate_image(
    uuid: UUID, service: ServiceDep, params: RenderParamsDep
) -> RenderedImage:
    return await service.render_diffraction_aggregate(uuid, params)


@router.get(
    '/{uuid}/patterns/{index}/image',
    response_model=RenderedImage,
    openapi_extra=openapi_operation(ResourceService.render_diffraction_pattern),
)
async def get_diffraction_pattern_image(
    uuid: UUID, index: int, service: ServiceDep, params: RenderParamsDep
) -> RenderedImage:
    return await service.render_diffraction_pattern(uuid, index, params)

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Query
from fastapi.responses import FileResponse

from ptychodus.api.processing import RunState
from ptychodus.api.scattering.geometry import ScatteringRegime
from ptychodus.api.storage import ResourceKind

from ptychodus.webapp.db.base import IngestState
from ptychodus.webapp.rendering import RenderedImage
from ptychodus.webapp.rendering.params import RenderParamsDep
from ptychodus.webapp.rendering.schemas import PlotImage
from ptychodus.webapp.routers.deps import ServiceDep, openapi_operation
from ptychodus.webapp.service.resources import ResourceService
from ptychodus.webapp.service.schemas import (
    CONVENTION_DESCRIPTION,
    EROSION_DIVISOR_DESCRIPTION,
    MAD_THRESHOLD_DESCRIPTION,
    REFERENCE_UUID_DESCRIPTION,
    RESAMPLE_TEST_DESCRIPTION,
    SNR_THRESHOLD_DESCRIPTION,
    ObjectComparisonRead,
    ObjectMetricsRead,
    Page,
    ProbeComparisonRead,
    ProbeMetricsRead,
    ProbePositionsComparisonRead,
    ProbePositionsMetricsRead,
    ProductRead,
    SpectralSNRConventionName,
)

router = APIRouter(prefix='/product', tags=['product'])

_REFERENCE_UUID = Query(description=REFERENCE_UUID_DESCRIPTION)
_EROSION_DIVISOR = Query(8, ge=1, description=EROSION_DIVISOR_DESCRIPTION)
_RESAMPLE_TEST = Query(True, description=RESAMPLE_TEST_DESCRIPTION)


@router.get(
    '',
    response_model=Page[ProductRead],
    openapi_extra=openapi_operation(ResourceService.list_product),
)
async def list_product(
    service: ServiceDep,
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
    return await service.list_product(
        limit=limit,
        offset=offset,
        derived_from_uuid=derived_from_uuid,
        ingest_state=ingest_state,
        run_state=run_state,
        photon_energy_eV_min=photon_energy_eV_min,
        photon_energy_eV_max=photon_energy_eV_max,
        scattering_regime=scattering_regime,
    )


@router.get(
    '/{uuid}',
    response_model=ProductRead,
    openapi_extra=openapi_operation(ResourceService.get_product),
)
async def get_product(uuid: UUID, service: ServiceDep) -> ProductRead:
    return await service.get_product(uuid)


@router.get(
    '/{uuid}/files/product',
    openapi_extra=openapi_operation(ResourceService.resource_file),
)
async def get_product_file(uuid: UUID, service: ServiceDep) -> FileResponse:
    """The product's current file: its final one, or the newest checkpoint while it runs."""
    path = await service.resource_file(ResourceKind.PRODUCT, uuid)
    return FileResponse(path, media_type='application/x-hdf5', filename=path.name)


@router.get(
    '/{uuid}/probe/image',
    response_model=RenderedImage,
    openapi_extra=openapi_operation(ResourceService.render_probe),
)
async def get_probe_image(
    uuid: UUID,
    service: ServiceDep,
    params: RenderParamsDep,
    incoherent: int = Query(0, ge=0, description='Incoherent probe mode index.'),
) -> RenderedImage:
    return await service.render_probe(uuid, incoherent, params)


@router.get(
    '/{uuid}/probe/modes/image',
    response_model=RenderedImage,
    openapi_extra=openapi_operation(ResourceService.render_probe_modes),
)
async def get_probe_modes_image(
    uuid: UUID, service: ServiceDep, params: RenderParamsDep
) -> RenderedImage:
    return await service.render_probe_modes(uuid, params)


@router.get(
    '/{uuid}/object/image',
    response_model=RenderedImage,
    openapi_extra=openapi_operation(ResourceService.render_object),
)
async def get_object_image(
    uuid: UUID, service: ServiceDep, params: RenderParamsDep
) -> RenderedImage:
    """Render the object as the product of its layers, the transmission of the whole stack."""
    return await service.render_object(uuid, params)


@router.get(
    '/{uuid}/object/{layer}/image',
    response_model=RenderedImage,
    openapi_extra=openapi_operation(ResourceService.render_object_layer),
)
async def get_object_layer_image(
    uuid: UUID, layer: int, service: ServiceDep, params: RenderParamsDep
) -> RenderedImage:
    return await service.render_object_layer(uuid, layer, params)


@router.get(
    '/positions/image',
    response_model=PlotImage,
    openapi_extra=openapi_operation(ResourceService.render_probe_positions),
)
async def get_positions_image(
    service: ServiceDep,
    uuid: Annotated[
        list[UUID],
        Query(
            min_length=1, description='Product to plot; repeat to overlay several on shared axes.'
        ),
    ],
    connect_path: bool = Query(True, description='Join successive probe positions with a line.'),
    width_px: int = Query(640, ge=128, le=2048),
    height_px: int = Query(640, ge=128, le=2048),
) -> PlotImage:
    """Plot one or more products' probe positions, one color and legend entry per product."""
    return await service.render_probe_positions(
        uuid, connect_path=connect_path, width_px=width_px, height_px=height_px
    )


@router.get(
    '/{uuid}/probe/metrics',
    response_model=ProbeMetricsRead,
    openapi_extra=openapi_operation(ResourceService.get_probe_metrics),
)
async def get_probe_metrics(uuid: UUID, service: ServiceDep) -> ProbeMetricsRead:
    """Describe the product's probe: its size, entropy, mode spectrum and scan overlap."""
    return await service.get_probe_metrics(uuid)


@router.get(
    '/{uuid}/object/metrics',
    response_model=ObjectMetricsRead,
    openapi_extra=openapi_operation(ResourceService.get_object_metrics),
)
async def get_object_metrics(
    uuid: UUID, service: ServiceDep, erosion_divisor: int = _EROSION_DIVISOR
) -> ObjectMetricsRead:
    """Describe the product's object over the region its scan illuminated."""
    return await service.get_object_metrics(uuid, erosion_divisor=erosion_divisor)


@router.get(
    '/{uuid}/positions/metrics',
    response_model=ProbePositionsMetricsRead,
    openapi_extra=openapi_operation(ResourceService.get_probe_positions_metrics),
)
async def get_probe_positions_metrics(uuid: UUID, service: ServiceDep) -> ProbePositionsMetricsRead:
    """Describe the product's probe positions: their extent, spacing, scan indexes and I0."""
    return await service.get_probe_positions_metrics(uuid)


@router.get(
    '/{uuid}/probe/comparison',
    response_model=ProbeComparisonRead,
    openapi_extra=openapi_operation(ResourceService.compare_probe),
)
async def compare_probe(
    uuid: UUID,
    service: ServiceDep,
    reference_uuid: UUID = _REFERENCE_UUID,
    erosion_divisor: int = _EROSION_DIVISOR,
    mad_threshold: float = Query(4.5, gt=0.0, description=MAD_THRESHOLD_DESCRIPTION),
    resample_test: bool = _RESAMPLE_TEST,
) -> ProbeComparisonRead:
    """Compare the product's probe with a reference product's."""
    return await service.compare_probe(
        uuid,
        reference_uuid,
        erosion_divisor=erosion_divisor,
        mad_threshold=mad_threshold,
        resample_test=resample_test,
    )


@router.get(
    '/{uuid}/object/comparison',
    response_model=ObjectComparisonRead,
    openapi_extra=openapi_operation(ResourceService.compare_object),
)
async def compare_object(
    uuid: UUID,
    service: ServiceDep,
    reference_uuid: UUID = _REFERENCE_UUID,
    convention: SpectralSNRConventionName = Query('half_split', description=CONVENTION_DESCRIPTION),
    snr_threshold: float = Query(2.0, gt=0.0, description=SNR_THRESHOLD_DESCRIPTION),
    erosion_divisor: int = _EROSION_DIVISOR,
    resample_test: bool = _RESAMPLE_TEST,
) -> ObjectComparisonRead:
    """Compare the product's projected object with a reference product's."""
    return await service.compare_object(
        uuid,
        reference_uuid,
        convention=convention,
        snr_threshold=snr_threshold,
        erosion_divisor=erosion_divisor,
        resample_test=resample_test,
    )


@router.get(
    '/{uuid}/positions/comparison',
    response_model=ProbePositionsComparisonRead,
    openapi_extra=openapi_operation(ResourceService.compare_probe_positions),
)
async def compare_probe_positions(
    uuid: UUID, service: ServiceDep, reference_uuid: UUID = _REFERENCE_UUID
) -> ProbePositionsComparisonRead:
    """Measure how the product's probe positions are displaced from a reference product's."""
    return await service.compare_probe_positions(uuid, reference_uuid)

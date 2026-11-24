from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Query
from fastapi.responses import FileResponse

from ptychodus.api.processing import RunState
from ptychodus.api.storage import ResourceKind

from ptychodus.webapp.db.base import IngestState
from ptychodus.webapp.rendering import RenderedImage
from ptychodus.webapp.rendering.params import RenderParamsDep
from ptychodus.webapp.routers.deps import ServiceDep, openapi_operation
from ptychodus.webapp.service.resources import ResourceService
from ptychodus.webapp.service.schemas import FluorescenceRead, Page

router = APIRouter(prefix='/fluorescence', tags=['fluorescence'])


@router.get(
    '',
    response_model=Page[FluorescenceRead],
    openapi_extra=openapi_operation(ResourceService.list_fluorescence),
)
async def list_fluorescence(
    service: ServiceDep,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    derived_from_uuid: UUID | None = None,
    element: str | None = None,
    ingest_state: IngestState | None = None,
    run_state: list[RunState] | None = Query(
        None, description='Only resources whose run is in one of these states; repeat for several.'
    ),
) -> Page[FluorescenceRead]:
    return await service.list_fluorescence(
        limit=limit,
        offset=offset,
        derived_from_uuid=derived_from_uuid,
        element=element,
        ingest_state=ingest_state,
        run_state=run_state,
    )


@router.get(
    '/{uuid}',
    response_model=FluorescenceRead,
    openapi_extra=openapi_operation(ResourceService.get_fluorescence),
)
async def get_fluorescence(uuid: UUID, service: ServiceDep) -> FluorescenceRead:
    return await service.get_fluorescence(uuid)


@router.get(
    '/{uuid}/files/fluorescence',
    openapi_extra=openapi_operation(ResourceService.resource_file),
)
async def get_fluorescence_file(uuid: UUID, service: ServiceDep) -> FileResponse:
    path = await service.resource_file(ResourceKind.FLUORESCENCE, uuid)
    return FileResponse(path, media_type='application/x-hdf5', filename=path.name)


_PAIRED_PRODUCT = Query(None, description='Optional paired product for physical pixel geometry.')


@router.get(
    '/{uuid}/image',
    response_model=RenderedImage,
    openapi_extra=openapi_operation(ResourceService.render_fluorescence_summary),
)
async def get_fluorescence_summary_image(
    uuid: UUID,
    service: ServiceDep,
    params: RenderParamsDep,
    product_uuid: UUID | None = _PAIRED_PRODUCT,
) -> RenderedImage:
    """Render the elementwise sum of every element map in the dataset."""
    return await service.render_fluorescence_summary(uuid, product_uuid, params)


@router.get(
    '/{uuid}/elements/{name}/image',
    response_model=RenderedImage,
    openapi_extra=openapi_operation(ResourceService.render_fluorescence_element),
)
async def get_fluorescence_element_image(
    uuid: UUID,
    name: str,
    service: ServiceDep,
    params: RenderParamsDep,
    product_uuid: UUID | None = _PAIRED_PRODUCT,
) -> RenderedImage:
    return await service.render_fluorescence_element(uuid, name, product_uuid, params)

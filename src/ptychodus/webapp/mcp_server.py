"""fastmcp server whose read-only tools are the service's operations, as the REST routes are."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Awaitable, Callable
from typing import Annotated, Any, TypeVar
from uuid import UUID

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.auth import AuthProvider
from fastmcp.server.dependencies import get_access_token
from fastmcp.server.middleware import CallNext, Middleware, MiddlewareContext
from fastmcp.tools import ToolResult
from mcp.types import CallToolRequestParams, ImageContent
from pydantic import BaseModel, Field

from ptychodus.api.diffraction import Polarization
from ptychodus.api.processing import RunState
from ptychodus.api.scattering.geometry import ScatteringRegime

from ptychodus.webapp.db.base import IngestState
from ptychodus.webapp.mcp_tools.annotations import READ_ONLY_ANNOTATIONS
from ptychodus.webapp.mcp_tools.xraydb import create_xraydb_mcp
from ptychodus.webapp.rendering import OptionsRead, RenderParams, coerce_render_params
from ptychodus.webapp.rendering.schemas import PlotImageInfo, RenderedImage, RenderedImageInfo
from ptychodus.webapp.service.errors import InvalidRequestError, ResourceNotFoundError
from ptychodus.webapp.service.operations import operation_class_of
from ptychodus.webapp.service.resources import ResourceService
from ptychodus.webapp.service.schemas import (
    CONVENTION_DESCRIPTION,
    EROSION_DIVISOR_DESCRIPTION,
    MAD_THRESHOLD_DESCRIPTION,
    REFERENCE_UUID_DESCRIPTION,
    RESAMPLE_TEST_DESCRIPTION,
    SNR_THRESHOLD_DESCRIPTION,
    CampaignRead,
    DiffractionRead,
    FluorescenceRead,
    LineageRead,
    ObjectComparisonRead,
    ObjectMetricsRead,
    Page,
    ProbeComparisonRead,
    ProbeMetricsRead,
    ProbePositionsComparisonRead,
    ProbePositionsMetricsRead,
    ProductRead,
    ResourceStats,
    SpectralSNRConventionName,
    nullable_numbers_schema,
)

__all__ = ['create_mcp_server']

logger = logging.getLogger(__name__)

_T = TypeVar('_T')

# The bounds the REST routes declare with Query(...), so both surfaces refuse alike.
_Limit = Annotated[int, Field(ge=1, le=200)]
_Offset = Annotated[int, Field(ge=0)]
_PlotSizePx = Annotated[int, Field(ge=128, le=2048)]
_ReferenceUUID = Annotated[UUID, Field(description=REFERENCE_UUID_DESCRIPTION)]
_ErosionDivisor = Annotated[int, Field(ge=1, description=EROSION_DIVISOR_DESCRIPTION)]
_ResampleTest = Annotated[bool, Field(description=RESAMPLE_TEST_DESCRIPTION)]
_MadThreshold = Annotated[float, Field(gt=0.0, description=MAD_THRESHOLD_DESCRIPTION)]
_Convention = Annotated[SpectralSNRConventionName, Field(description=CONVENTION_DESCRIPTION)]
_SNRThreshold = Annotated[float, Field(gt=0.0, description=SNR_THRESHOLD_DESCRIPTION)]


def _tool_error(exc: ResourceNotFoundError | InvalidRequestError) -> ToolError:
    """The refusal as the tool states it: REST's detail message, then any choices it lists."""
    if isinstance(exc, ResourceNotFoundError):
        if exc.available:
            return ToolError(f'{exc.message}; available: [{", ".join(exc.available)}]')

        return ToolError(exc.message)

    choices = [
        f'{key}: [{", ".join(values)}]'
        for key, values in (exc.detail or {}).items()
        if key != 'error'
    ]
    return ToolError('; '.join([exc.message, *choices]))


async def _served(operation: Awaitable[_T]) -> _T:
    """Await a service operation, stating its refusal as a :class:`ToolError`."""
    try:
        return await operation
    except (ResourceNotFoundError, InvalidRequestError) as exc:
        raise _tool_error(exc) from exc


def _tool_options(method: Callable[..., Any], title: str) -> dict[str, Any]:
    """A tool's title, read-only annotations and description, the last from `method`.

    The description is the service method's docstring followed by its operation class.
    """
    op_class = operation_class_of(method)
    description = f'{inspect.getdoc(method)}\n\nOperation class: {op_class.value}.'
    annotations = READ_ONLY_ANNOTATIONS.model_copy(update={'title': title})
    return {'title': title, 'description': description, 'annotations': annotations}


def _render_params(
    colormap: str,
    transform: str,
    component: str | None,
    color_model: str | None,
    value_min: float | None,
    value_max: float | None,
    clip: bool,
) -> RenderParams:
    """Coerce raw render parameters, listing the valid choices REST's 400 detail lists."""
    try:
        return coerce_render_params(
            colormap=colormap,
            transform=transform,
            component=component,
            color_model=color_model,
            value_min=value_min,
            value_max=value_max,
            clip=clip,
        )
    except InvalidRequestError as exc:
        raise _tool_error(exc) from exc


def _image_result(png_base64: str, info: BaseModel) -> ToolResult:
    """The PNG as image content beside `info` as the structured result."""
    image = ImageContent(type='image', data=png_base64, mime_type='image/png')
    return ToolResult(content=[image], structured_content=info.model_dump(mode='json'))


def _rendered_result(rendered: RenderedImage) -> ToolResult:
    return _image_result(
        rendered.png_base64, RenderedImageInfo.model_validate(rendered.model_dump())
    )


class _CallLog(Middleware):
    """Logs each tool call with the client its bearer token names."""

    async def on_call_tool(
        self,
        context: MiddlewareContext[CallToolRequestParams],
        call_next: CallNext[CallToolRequestParams, ToolResult],
    ) -> ToolResult:
        token = get_access_token()
        client_id = token.client_id if token is not None else '(unauthenticated)'
        logger.info('tool %s called by %s', context.message.name, client_id)
        return await call_next(context)


def create_mcp_server(service: ResourceService, *, auth: AuthProvider | None = None) -> FastMCP:
    """The MCP server over `service`, verifying bearer tokens with `auth` when one is given."""
    instructions = """\
This server reads a ptychodus storage root: the x-ray ptychography data an instrument \
recorded and the reconstructions made from it. Every tool is read-only.

Resources. Each resource has a uuid and a kind. A campaign groups diffraction datasets, \
which hold the measured diffraction patterns, one per probe position, paired with the \
positions by scan index. A product holds a probe (the complex illumination function, in \
photons per pixel), an object (the complex transmission function of the sample) and the \
probe positions, with their metadata; it is either an initial guess or a reconstruction. \
A fluorescence dataset holds element maps. A resource's derived_from edges name the \
resources it was made from; get_lineage walks them both ways.

Runs. A resource a process produced carries a run record: state (pending, running, \
succeeded, failed, cancelled), reason, and progress against progress_goal (epochs for a \
reconstruction; 0 when unknown). Imported data has none. A running product's data is its \
newest checkpoint and changes as the run advances; list tools filter by run_state.

Units. Quantities are SI and their names carry the unit: _m, _eV, _s, _deg for geometric \
angles, _rad for phases, _px for pixels. I0 is the per-position incident-flux reading.

Images. Call get_visualization_options for the valid colormaps, transforms, complex \
components and color models. A render_* tool returns a PNG with its colour scale, value \
label and pixel size as structured content.

Analysis. get_probe_metrics, get_object_metrics and get_probe_positions_metrics describe \
one product; compare_probe, compare_object and compare_probe_positions compare a product \
with a reference product. They read and compute on the product data, which can take \
tens of seconds for a large product. A null statistic is one the report cannot define, \
such as I0 on a product without it.

Operation classes. Each tool's description ends with its class: instant tools read the \
metadata cache; bounded tools also read and compute on resource data.

Links. The web UI shows a resource at /ui/#<kind>/<uuid>, and a product's probe, object \
or probe positions at /ui/#product/<uuid>/probe, /object or /positions.
"""
    mcp = FastMCP(
        name='ptychodus-webapp', instructions=instructions, auth=auth, middleware=[_CallLog()]
    )

    @mcp.tool(**_tool_options(service.list_campaign, 'List campaigns'))
    async def list_campaign(
        limit: _Limit = 50,
        offset: _Offset = 0,
        sample_name: str | None = None,
        tag: str | None = None,
        ingest_state: IngestState | None = None,
    ) -> Page[CampaignRead]:
        return await _served(
            service.list_campaign(
                limit=limit,
                offset=offset,
                sample_name=sample_name,
                tag=tag,
                ingest_state=ingest_state,
            )
        )

    @mcp.tool(**_tool_options(service.get_campaign, 'Get a campaign'))
    async def get_campaign(uuid: UUID) -> CampaignRead:
        return await _served(service.get_campaign(uuid))

    @mcp.tool(**_tool_options(service.list_diffraction, 'List diffraction datasets'))
    async def list_diffraction(
        limit: _Limit = 50,
        offset: _Offset = 0,
        campaign_uuid: UUID | None = None,
        derived_from_uuid: UUID | None = None,
        ingest_state: IngestState | None = None,
        run_state: list[RunState] | None = None,
        photon_energy_eV_min: float | None = None,  # noqa: N803
        photon_energy_eV_max: float | None = None,  # noqa: N803
        tomography_angle_deg_min: float | None = None,
        tomography_angle_deg_max: float | None = None,
        tilt_angle_deg_min: float | None = None,
        tilt_angle_deg_max: float | None = None,
        polarization: Polarization | None = None,
    ) -> Page[DiffractionRead]:
        return await _served(
            service.list_diffraction(
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
        )

    @mcp.tool(**_tool_options(service.get_diffraction, 'Get a diffraction dataset'))
    async def get_diffraction(uuid: UUID) -> DiffractionRead:
        return await _served(service.get_diffraction(uuid))

    @mcp.tool(**_tool_options(service.list_product, 'List products'))
    async def list_product(
        limit: _Limit = 50,
        offset: _Offset = 0,
        derived_from_uuid: UUID | None = None,
        ingest_state: IngestState | None = None,
        run_state: list[RunState] | None = None,
        photon_energy_eV_min: float | None = None,  # noqa: N803
        photon_energy_eV_max: float | None = None,  # noqa: N803
        scattering_regime: ScatteringRegime | None = None,
    ) -> Page[ProductRead]:
        return await _served(
            service.list_product(
                limit=limit,
                offset=offset,
                derived_from_uuid=derived_from_uuid,
                ingest_state=ingest_state,
                run_state=run_state,
                photon_energy_eV_min=photon_energy_eV_min,
                photon_energy_eV_max=photon_energy_eV_max,
                scattering_regime=scattering_regime,
            )
        )

    @mcp.tool(**_tool_options(service.get_product, 'Get a product'))
    async def get_product(uuid: UUID) -> ProductRead:
        return await _served(service.get_product(uuid))

    @mcp.tool(**_tool_options(service.list_fluorescence, 'List fluorescence datasets'))
    async def list_fluorescence(
        limit: _Limit = 50,
        offset: _Offset = 0,
        derived_from_uuid: UUID | None = None,
        element: str | None = None,
        ingest_state: IngestState | None = None,
        run_state: list[RunState] | None = None,
    ) -> Page[FluorescenceRead]:
        return await _served(
            service.list_fluorescence(
                limit=limit,
                offset=offset,
                derived_from_uuid=derived_from_uuid,
                element=element,
                ingest_state=ingest_state,
                run_state=run_state,
            )
        )

    @mcp.tool(**_tool_options(service.get_fluorescence, 'Get a fluorescence dataset'))
    async def get_fluorescence(uuid: UUID) -> FluorescenceRead:
        return await _served(service.get_fluorescence(uuid))

    @mcp.tool(**_tool_options(service.get_lineage, 'Get lineage'))
    async def get_lineage(uuid: UUID) -> LineageRead:
        return await _served(service.get_lineage(uuid))

    @mcp.tool(**_tool_options(service.get_resource_stats, 'Get resource counts'))
    async def get_resource_stats() -> ResourceStats:
        return await _served(service.get_resource_stats())

    @mcp.tool(**_tool_options(service.get_visualization_options, 'Get visualization options'))
    async def get_visualization_options() -> OptionsRead:
        return await _served(service.get_visualization_options())

    @mcp.tool(
        output_schema=RenderedImageInfo.model_json_schema(),
        **_tool_options(service.render_diffraction_pattern, 'Render a diffraction pattern'),
    )
    async def render_diffraction_pattern(
        uuid: UUID,
        index: int,
        colormap: str = 'gray',
        transform: str = 'identity',
        value_min: float | None = None,
        value_max: float | None = None,
        clip: bool = False,
    ) -> ToolResult:
        params = _render_params(colormap, transform, None, None, value_min, value_max, clip)
        rendered = await _served(service.render_diffraction_pattern(uuid, index, params))
        return _rendered_result(rendered)

    @mcp.tool(
        output_schema=RenderedImageInfo.model_json_schema(),
        **_tool_options(service.render_diffraction_aggregate, 'Render the mean pattern'),
    )
    async def render_diffraction_aggregate(
        uuid: UUID,
        colormap: str = 'gray',
        transform: str = 'identity',
        value_min: float | None = None,
        value_max: float | None = None,
        clip: bool = False,
    ) -> ToolResult:
        params = _render_params(colormap, transform, None, None, value_min, value_max, clip)
        rendered = await _served(service.render_diffraction_aggregate(uuid, params))
        return _rendered_result(rendered)

    @mcp.tool(
        output_schema=RenderedImageInfo.model_json_schema(),
        **_tool_options(service.render_probe, 'Render a probe mode'),
    )
    async def render_probe(
        uuid: UUID,
        incoherent: Annotated[int, Field(ge=0)] = 0,
        colormap: str = 'gray',
        transform: str = 'identity',
        component: str | None = None,
        color_model: str | None = None,
        value_min: float | None = None,
        value_max: float | None = None,
        clip: bool = False,
    ) -> ToolResult:
        params = _render_params(
            colormap, transform, component, color_model, value_min, value_max, clip
        )
        rendered = await _served(service.render_probe(uuid, incoherent, params))
        return _rendered_result(rendered)

    @mcp.tool(
        output_schema=RenderedImageInfo.model_json_schema(),
        **_tool_options(service.render_probe_modes, 'Render all probe modes'),
    )
    async def render_probe_modes(
        uuid: UUID,
        colormap: str = 'gray',
        transform: str = 'identity',
        component: str | None = None,
        color_model: str | None = None,
        value_min: float | None = None,
        value_max: float | None = None,
        clip: bool = False,
    ) -> ToolResult:
        params = _render_params(
            colormap, transform, component, color_model, value_min, value_max, clip
        )
        rendered = await _served(service.render_probe_modes(uuid, params))
        return _rendered_result(rendered)

    @mcp.tool(
        output_schema=RenderedImageInfo.model_json_schema(),
        **_tool_options(service.render_object, 'Render the object'),
    )
    async def render_object(
        uuid: UUID,
        colormap: str = 'gray',
        transform: str = 'identity',
        component: str | None = None,
        color_model: str | None = None,
        value_min: float | None = None,
        value_max: float | None = None,
        clip: bool = False,
    ) -> ToolResult:
        params = _render_params(
            colormap, transform, component, color_model, value_min, value_max, clip
        )
        rendered = await _served(service.render_object(uuid, params))
        return _rendered_result(rendered)

    @mcp.tool(
        output_schema=RenderedImageInfo.model_json_schema(),
        **_tool_options(service.render_object_layer, 'Render an object layer'),
    )
    async def render_object_layer(
        uuid: UUID,
        layer: int,
        colormap: str = 'gray',
        transform: str = 'identity',
        component: str | None = None,
        color_model: str | None = None,
        value_min: float | None = None,
        value_max: float | None = None,
        clip: bool = False,
    ) -> ToolResult:
        params = _render_params(
            colormap, transform, component, color_model, value_min, value_max, clip
        )
        rendered = await _served(service.render_object_layer(uuid, layer, params))
        return _rendered_result(rendered)

    @mcp.tool(
        output_schema=PlotImageInfo.model_json_schema(),
        **_tool_options(service.render_probe_positions, 'Plot probe positions'),
    )
    async def render_probe_positions(
        uuid: Annotated[list[UUID], Field(min_length=1)],
        connect_path: bool = True,
        width_px: _PlotSizePx = 640,
        height_px: _PlotSizePx = 640,
    ) -> ToolResult:
        plot = await _served(
            service.render_probe_positions(
                uuid, connect_path=connect_path, width_px=width_px, height_px=height_px
            )
        )
        return _image_result(plot.png_base64, PlotImageInfo.model_validate(plot.model_dump()))

    @mcp.tool(
        output_schema=RenderedImageInfo.model_json_schema(),
        **_tool_options(service.render_fluorescence_summary, 'Render all element maps summed'),
    )
    async def render_fluorescence_summary(
        uuid: UUID,
        product_uuid: UUID | None = None,
        colormap: str = 'gray',
        transform: str = 'identity',
        value_min: float | None = None,
        value_max: float | None = None,
        clip: bool = False,
    ) -> ToolResult:
        params = _render_params(colormap, transform, None, None, value_min, value_max, clip)
        rendered = await _served(service.render_fluorescence_summary(uuid, product_uuid, params))
        return _rendered_result(rendered)

    @mcp.tool(
        output_schema=RenderedImageInfo.model_json_schema(),
        **_tool_options(service.render_fluorescence_element, 'Render an element map'),
    )
    async def render_fluorescence_element(
        uuid: UUID,
        name: str,
        product_uuid: UUID | None = None,
        colormap: str = 'gray',
        transform: str = 'identity',
        value_min: float | None = None,
        value_max: float | None = None,
        clip: bool = False,
    ) -> ToolResult:
        params = _render_params(colormap, transform, None, None, value_min, value_max, clip)
        rendered = await _served(
            service.render_fluorescence_element(uuid, name, product_uuid, params)
        )
        return _rendered_result(rendered)

    @mcp.tool(
        output_schema=nullable_numbers_schema(ProbeMetricsRead),
        **_tool_options(service.get_probe_metrics, 'Get probe metrics'),
    )
    async def get_probe_metrics(uuid: UUID) -> ProbeMetricsRead:
        return await _served(service.get_probe_metrics(uuid))

    @mcp.tool(
        output_schema=nullable_numbers_schema(ObjectMetricsRead),
        **_tool_options(service.get_object_metrics, 'Get object metrics'),
    )
    async def get_object_metrics(
        uuid: UUID, erosion_divisor: _ErosionDivisor = 8
    ) -> ObjectMetricsRead:
        return await _served(service.get_object_metrics(uuid, erosion_divisor=erosion_divisor))

    @mcp.tool(
        output_schema=nullable_numbers_schema(ProbePositionsMetricsRead),
        **_tool_options(service.get_probe_positions_metrics, 'Get probe-position metrics'),
    )
    async def get_probe_positions_metrics(uuid: UUID) -> ProbePositionsMetricsRead:
        return await _served(service.get_probe_positions_metrics(uuid))

    @mcp.tool(
        output_schema=nullable_numbers_schema(ProbeComparisonRead),
        **_tool_options(service.compare_probe, 'Compare probes'),
    )
    async def compare_probe(
        uuid: UUID,
        reference_uuid: _ReferenceUUID,
        erosion_divisor: _ErosionDivisor = 8,
        mad_threshold: _MadThreshold = 4.5,
        resample_test: _ResampleTest = True,
    ) -> ProbeComparisonRead:
        return await _served(
            service.compare_probe(
                uuid,
                reference_uuid,
                erosion_divisor=erosion_divisor,
                mad_threshold=mad_threshold,
                resample_test=resample_test,
            )
        )

    @mcp.tool(
        output_schema=nullable_numbers_schema(ObjectComparisonRead),
        **_tool_options(service.compare_object, 'Compare objects'),
    )
    async def compare_object(
        uuid: UUID,
        reference_uuid: _ReferenceUUID,
        convention: _Convention = 'half_split',
        snr_threshold: _SNRThreshold = 2.0,
        erosion_divisor: _ErosionDivisor = 8,
        resample_test: _ResampleTest = True,
    ) -> ObjectComparisonRead:
        return await _served(
            service.compare_object(
                uuid,
                reference_uuid,
                convention=convention,
                snr_threshold=snr_threshold,
                erosion_divisor=erosion_divisor,
                resample_test=resample_test,
            )
        )

    @mcp.tool(
        output_schema=nullable_numbers_schema(ProbePositionsComparisonRead),
        **_tool_options(service.compare_probe_positions, 'Compare probe positions'),
    )
    async def compare_probe_positions(
        uuid: UUID, reference_uuid: _ReferenceUUID
    ) -> ProbePositionsComparisonRead:
        return await _served(service.compare_probe_positions(uuid, reference_uuid))

    mcp.mount(create_xraydb_mcp(), namespace='xraydb')

    return mcp

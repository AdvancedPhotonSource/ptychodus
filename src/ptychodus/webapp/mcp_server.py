"""fastmcp server with read-only tools mirroring the REST surface."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Annotated, Any, TypeVar
from uuid import UUID

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.tools import ToolResult
from mcp.types import ImageContent
from pydantic import BaseModel, Field
from sqlalchemy import String, exists, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ptychodus.api.diffraction import Polarization
from ptychodus.api.processing import RunState
from ptychodus.api.scattering.geometry import ScatteringRegime
from ptychodus.api.storage import (
    LayoutError,
    ManifestLoadError,
    ResourceKind,
    StorageLayout,
    resolve_resource_file,
)
from ptychodus.api.visualize import VisualizationProduct

from ptychodus.webapp.db import repositories as repo
from ptychodus.webapp.db.base import IngestState
from ptychodus.webapp.db.models import Campaign, DerivationEdge, Diffraction, Fluorescence, Product
from ptychodus.webapp.db.session import SessionProvider
from ptychodus.webapp.mcp_tools.xraydb import create_xraydb_mcp
from ptychodus.webapp.rendering import (
    OptionsRead,
    RenderParams,
    RenderParamsError,
    coerce_render_params,
    visualization_options,
)
from ptychodus.webapp.rendering.offload import RenderRunner
from ptychodus.webapp.rendering.params import InvalidRenderParamError
from ptychodus.webapp.rendering.schemas import PlotImageInfo, RenderedImageInfo
from ptychodus.webapp.rendering.views import (
    RenderLookupError,
    diffraction_aggregate_view,
    diffraction_pattern_view,
    fluorescence_element_view,
    fluorescence_summary_view,
    object_layer_view,
    object_view,
    probe_modes_view,
    probe_view,
    scan_paths_view,
    view_as_rendered_image,
)
from ptychodus.webapp.routers._convert import (
    campaign_to_read,
    diffraction_to_read,
    fluorescence_to_read,
    product_to_read,
    row_label,
)
from ptychodus.webapp.routers.schemas import (
    CampaignRead,
    DiffractionRead,
    FluorescenceRead,
    LineageNode,
    LineageRead,
    Page,
    ProductRead,
    StoreStats,
)

logger = logging.getLogger(__name__)

_T = TypeVar('_T')

# The bounds the REST routes declare with Query(...), so both surfaces refuse alike.
_Limit = Annotated[int, Field(ge=1, le=200)]
_Offset = Annotated[int, Field(ge=0)]
_PlotSizePx = Annotated[int, Field(ge=128, le=2048)]

_RENDERED_IMAGE_SCHEMA = RenderedImageInfo.model_json_schema()
_PLOT_IMAGE_SCHEMA = PlotImageInfo.model_json_schema()


class _Ctx:
    provider: SessionProvider | None = None
    layout: StorageLayout | None = None
    render_runner: RenderRunner | None = None


def bind_session_provider(provider: SessionProvider) -> None:
    _Ctx.provider = provider


def bind_layout(layout: StorageLayout) -> None:
    _Ctx.layout = layout


def bind_render_runner(runner: RenderRunner) -> None:
    _Ctx.render_runner = runner


@asynccontextmanager
async def _session() -> AsyncIterator[AsyncSession]:
    if _Ctx.provider is None:
        raise RuntimeError('SessionProvider not bound; call bind_session_provider() first')
    async with _Ctx.provider.session_factory() as session:
        yield session


def _require_layout() -> StorageLayout:
    if _Ctx.layout is None:
        raise ToolError('StorageLayout not bound; call bind_layout() first')
    return _Ctx.layout


async def _resource_file(kind: str, uuid: UUID) -> Path:
    """The file holding the resource's current data, as its manifest names it.

    A running resource names its newest checkpoint, and one with no result yet has none.
    Raises :class:`ToolError` with the REST route's 404 detail when either is missing.
    """
    async with _session() as session:
        row = await repo.get_row_by_kind(session, kind, uuid)

    if row is None:
        raise ToolError(f'{kind} {uuid} not found')

    try:
        path = resolve_resource_file(_require_layout(), uuid, kind)
    except (LayoutError, ManifestLoadError) as exc:
        raise ToolError(str(exc)) from exc

    if not path.is_file():
        raise ToolError(f'{path.name} not present on disk')

    return path


async def _paired_product_file(product_uuid: UUID | None) -> Path | None:
    """The paired product whose object pixel geometry a map takes, if one was named."""
    if product_uuid is None:
        return None

    async with _session() as session:
        row = await repo.get_row(session, Product, product_uuid)

    if row is None:
        raise ToolError(f'product {product_uuid} not found for pixel geometry')

    return await _resource_file(ResourceKind.PRODUCT, product_uuid)


async def _render(func: Callable[..., _T], /, *args: Any) -> _T:
    """Run a render in a worker thread, mapping its errors to :class:`ToolError`."""
    if _Ctx.render_runner is None:
        raise ToolError('RenderRunner not bound; call bind_render_runner() first')

    try:
        return await _Ctx.render_runner.run(func, *args)
    except RenderLookupError as exc:
        if exc.available:
            raise ToolError(f'{exc.message}; available: [{", ".join(exc.available)}]') from exc

        raise ToolError(exc.message) from exc
    except RenderParamsError as exc:
        raise ToolError(str(exc)) from exc


def _image_result(png_base64: str, info: BaseModel) -> ToolResult:
    """The PNG as image content beside `info` as the structured result."""
    image = ImageContent(type='image', data=png_base64, mime_type='image/png')
    return ToolResult(content=[image], structured_content=info.model_dump(mode='json'))


async def _render_image(
    view: Callable[..., VisualizationProduct],
    arguments: Sequence[Any],
    params: RenderParams,
    is_complex: bool,
) -> ToolResult:
    """Render `view` as the REST route does, returned as the image and its scales."""
    rendered = await _render(view_as_rendered_image, view, arguments, params, is_complex)
    return _image_result(
        rendered.png_base64, RenderedImageInfo.model_validate(rendered.model_dump())
    )


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
    except InvalidRenderParamError as exc:
        choices = [
            f'{key}: [{", ".join(values)}]' for key, values in exc.detail.items() if key != 'error'
        ]
        raise ToolError('; '.join([str(exc), *choices])) from exc


def create_mcp_server() -> FastMCP:
    mcp = FastMCP(name='ptychodus-webapp')

    @mcp.tool()
    async def list_campaign(
        limit: _Limit = 50,
        offset: _Offset = 0,
        sample_name: str | None = None,
        tag: str | None = None,
        ingest_state: IngestState | None = None,
    ) -> Page[CampaignRead]:
        """List campaigns, optionally only those carrying `tag`."""
        async with _session() as session:
            where = []
            if sample_name is not None:
                where.append(Campaign.sample_name == sample_name)
            if ingest_state is not None:
                where.append(Campaign.ingest_state == ingest_state)
            if tag is not None:
                # tags is a JSON-encoded TEXT column; match the quoted JSON form of the value.
                where.append(func.instr(func.cast(Campaign.tags, String), f'"{tag}"') > 0)
            items, total = await repo.list_rows(
                session, Campaign, limit=limit, offset=offset, where=where
            )
            return Page(
                items=[campaign_to_read(i) for i in items],
                total=total,
                limit=limit,
                offset=offset,
            )

    @mcp.tool()
    async def get_campaign(uuid: UUID) -> CampaignRead:
        """Get a campaign by UUID."""
        async with _session() as session:
            row = await repo.get_row(session, Campaign, uuid)
            if row is None:
                raise ToolError(f'campaign {uuid} not found')
            return campaign_to_read(row)

    @mcp.tool()
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
        """List diffraction datasets.

        `run_state` keeps only resources whose run is in one of those states (pending, running,
        succeeded, failed, cancelled); imported data has no run.
        """
        async with _session() as session:
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

    @mcp.tool()
    async def get_diffraction(uuid: UUID) -> DiffractionRead:
        """Get a diffraction dataset by UUID."""
        async with _session() as session:
            row = await repo.get_row(session, Diffraction, uuid)
            if row is None:
                raise ToolError(f'diffraction {uuid} not found')
            return await diffraction_to_read(session, row)

    @mcp.tool()
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
        """List reconstruction products, optionally only transmission, Bragg or Laue ones.

        `run_state` keeps only resources whose run is in one of those states (pending, running,
        succeeded, failed, cancelled); imported data has no run.
        """
        async with _session() as session:
            where = []
            if scattering_regime is not None:
                where.append(Product.scattering_regime == scattering_regime.value)
            if ingest_state is not None:
                where.append(Product.ingest_state == ingest_state)
            if run_state:
                where.append(Product.run_state.in_([s.value for s in run_state]))
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
            items, total = await repo.list_rows(
                session, Product, limit=limit, offset=offset, where=where
            )
            reads = [await product_to_read(session, i) for i in items]
            return Page(items=reads, total=total, limit=limit, offset=offset)

    @mcp.tool()
    async def get_product(uuid: UUID) -> ProductRead:
        """Get a product by UUID."""
        async with _session() as session:
            row = await repo.get_row(session, Product, uuid)
            if row is None:
                raise ToolError(f'product {uuid} not found')
            return await product_to_read(session, row)

    @mcp.tool()
    async def list_fluorescence(
        limit: _Limit = 50,
        offset: _Offset = 0,
        derived_from_uuid: UUID | None = None,
        element: str | None = None,
        ingest_state: IngestState | None = None,
        run_state: list[RunState] | None = None,
    ) -> Page[FluorescenceRead]:
        """List fluorescence datasets.

        `run_state` keeps only resources whose run is in one of those states (pending, running,
        succeeded, failed, cancelled); imported data has no run.
        """
        async with _session() as session:
            where = []
            if ingest_state is not None:
                where.append(Fluorescence.ingest_state == ingest_state)
            if run_state:
                where.append(Fluorescence.run_state.in_([s.value for s in run_state]))
            if element is not None:
                where.append(
                    func.instr(func.cast(Fluorescence.element_names, String), f'"{element}"') > 0
                )
            if derived_from_uuid is not None:
                edge_subq = select(DerivationEdge.source_uuid).where(
                    DerivationEdge.source_uuid == Fluorescence.uuid,
                    DerivationEdge.target_uuid == derived_from_uuid,
                )
                where.append(exists(edge_subq))
            items, total = await repo.list_rows(
                session, Fluorescence, limit=limit, offset=offset, where=where
            )
            reads = [await fluorescence_to_read(session, i) for i in items]
            return Page(items=reads, total=total, limit=limit, offset=offset)

    @mcp.tool()
    async def get_fluorescence(uuid: UUID) -> FluorescenceRead:
        """Get a fluorescence dataset by UUID."""
        async with _session() as session:
            row = await repo.get_row(session, Fluorescence, uuid)
            if row is None:
                raise ToolError(f'fluorescence {uuid} not found')
            return await fluorescence_to_read(session, row)

    @mcp.tool()
    async def get_lineage(uuid: UUID) -> LineageRead:
        """Walk the derivation DAG up and down from a node."""
        from ptychodus.webapp.routers.lineage import (
            _find_campaign,
            _resolve_node,
            _walk_ancestors,
            _walk_descendants,
        )

        async with _session() as session:
            resolved = await _resolve_node(session, uuid)
            if resolved is None:
                raise ToolError(f'no resource with uuid {uuid}')
            kind, row = resolved
            node = LineageNode(kind=kind, uuid=uuid, label=row_label(row))
            ancestors = await _walk_ancestors(session, uuid)
            descendants = await _walk_descendants(session, uuid)
            campaign = await _find_campaign(session, uuid, ancestors)
            return LineageRead(
                node=node, ancestors=ancestors, descendants=descendants, campaign=campaign
            )

    @mcp.tool()
    async def get_store_stats() -> StoreStats:
        """Return resource counts and invalid-row count for the store."""
        async with _session() as session:

            async def _count(model):
                return int(
                    (await session.execute(select(func.count()).select_from(model))).scalar_one()
                )

            invalid_total = 0
            for model in (Campaign, Diffraction, Product, Fluorescence):
                stmt = (
                    select(func.count())
                    .select_from(model)
                    .where(
                        model.ingest_state.in_(
                            [IngestState.INVALID, IngestState.MISSING_FILES, IngestState.ORPHANED]
                        )
                    )
                )
                invalid_total += int((await session.execute(stmt)).scalar_one())

            return StoreStats(
                campaign_count=await _count(Campaign),
                diffraction_count=await _count(Diffraction),
                product_count=await _count(Product),
                fluorescence_count=await _count(Fluorescence),
                invalid_count=invalid_total,
            )

    @mcp.tool()
    async def get_visualization_options() -> OptionsRead:
        """Enumerate valid choices for colormap / transform / component / color_model."""
        return visualization_options()

    @mcp.tool(output_schema=_RENDERED_IMAGE_SCHEMA)
    async def render_diffraction_pattern(
        uuid: UUID,
        index: int,
        colormap: str = 'gray',
        transform: str = 'identity',
        value_min: float | None = None,
        value_max: float | None = None,
        clip: bool = False,
    ) -> ToolResult:
        """Render a single diffraction pattern from an assembled dataset, with its scales."""
        params = _render_params(colormap, transform, None, None, value_min, value_max, clip)
        path = await _resource_file(ResourceKind.DIFFRACTION, uuid)
        return await _render_image(diffraction_pattern_view, (path, index), params, False)

    @mcp.tool(output_schema=_RENDERED_IMAGE_SCHEMA)
    async def render_diffraction_aggregate(
        uuid: UUID,
        colormap: str = 'gray',
        transform: str = 'identity',
        value_min: float | None = None,
        value_max: float | None = None,
        clip: bool = False,
    ) -> ToolResult:
        """Render the mean pattern across an assembled diffraction dataset, with its scales."""
        params = _render_params(colormap, transform, None, None, value_min, value_max, clip)
        path = await _resource_file(ResourceKind.DIFFRACTION, uuid)
        return await _render_image(diffraction_aggregate_view, (path,), params, False)

    @mcp.tool(output_schema=_RENDERED_IMAGE_SCHEMA)
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
        """Render a single incoherent probe mode, with its scales. The coherent axis is 0."""
        params = _render_params(
            colormap, transform, component, color_model, value_min, value_max, clip
        )
        path = await _resource_file(ResourceKind.PRODUCT, uuid)
        return await _render_image(probe_view, (path, incoherent), params, True)

    @mcp.tool(output_schema=_RENDERED_IMAGE_SCHEMA)
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
        """Render all incoherent probe modes tiled horizontally, with their scales."""
        params = _render_params(
            colormap, transform, component, color_model, value_min, value_max, clip
        )
        path = await _resource_file(ResourceKind.PRODUCT, uuid)
        return await _render_image(probe_modes_view, (path,), params, True)

    @mcp.tool(output_schema=_RENDERED_IMAGE_SCHEMA)
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
        """Render the object as the product of its layers, with its scales."""
        params = _render_params(
            colormap, transform, component, color_model, value_min, value_max, clip
        )
        path = await _resource_file(ResourceKind.PRODUCT, uuid)
        return await _render_image(object_view, (path,), params, True)

    @mcp.tool(output_schema=_RENDERED_IMAGE_SCHEMA)
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
        """Render a single object layer as a complex-valued image, with its scales."""
        params = _render_params(
            colormap, transform, component, color_model, value_min, value_max, clip
        )
        path = await _resource_file(ResourceKind.PRODUCT, uuid)
        return await _render_image(object_layer_view, (path, layer), params, True)

    @mcp.tool(output_schema=_PLOT_IMAGE_SCHEMA)
    async def render_scan_positions(
        uuid: Annotated[list[UUID], Field(min_length=1)],
        connect_path: bool = True,
        width_px: _PlotSizePx = 640,
        height_px: _PlotSizePx = 640,
    ) -> ToolResult:
        """Plot scan paths, one color and legend entry per product, with the plot's size.

        Pass several UUIDs to overlay them on shared axes and compare scans.
        """
        entries = [
            (await _resource_file(ResourceKind.PRODUCT, product_uuid), str(product_uuid)[:8])
            for product_uuid in uuid
        ]
        plot = await _render(
            partial(
                scan_paths_view, connect_path=connect_path, width_px=width_px, height_px=height_px
            ),
            entries,
        )
        return _image_result(plot.png_base64, PlotImageInfo.model_validate(plot.model_dump()))

    @mcp.tool(output_schema=_RENDERED_IMAGE_SCHEMA)
    async def render_fluorescence_summary(
        uuid: UUID,
        product_uuid: UUID | None = None,
        colormap: str = 'gray',
        transform: str = 'identity',
        value_min: float | None = None,
        value_max: float | None = None,
        clip: bool = False,
    ) -> ToolResult:
        """Render the elementwise sum of every element map in a dataset, with its scales."""
        params = _render_params(colormap, transform, None, None, value_min, value_max, clip)
        path = await _resource_file(ResourceKind.FLUORESCENCE, uuid)
        product_path = await _paired_product_file(product_uuid)
        return await _render_image(fluorescence_summary_view, (path, product_path), params, False)

    @mcp.tool(output_schema=_RENDERED_IMAGE_SCHEMA)
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
        """Render one element map from a fluorescence dataset, with its scales."""
        params = _render_params(colormap, transform, None, None, value_min, value_max, clip)
        path = await _resource_file(ResourceKind.FLUORESCENCE, uuid)
        product_path = await _paired_product_file(product_uuid)
        return await _render_image(
            fluorescence_element_view, (path, name, product_path), params, False
        )

    mcp.mount(create_xraydb_mcp(), namespace='xraydb')

    return mcp

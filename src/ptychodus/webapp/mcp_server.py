"""fastmcp server with read-only tools mirroring the REST surface."""

from __future__ import annotations

import logging
from base64 import b64decode
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Any, TypeVar
from uuid import UUID

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.utilities.types import Image as MCPImage
from sqlalchemy import String, func, select
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

from ptychodus.webapp.db import repositories as repo
from ptychodus.webapp.db.base import IngestState
from ptychodus.webapp.db.models import Campaign, DerivationEdge, Diffraction, Fluorescence, Product
from ptychodus.webapp.db.session import SessionProvider
from ptychodus.webapp.mcp_tools.xraydb import create_xraydb_mcp
from ptychodus.webapp.rendering import (
    OptionsRead,
    RenderParamsError,
    coerce_render_params,
    visualization_options,
)
from ptychodus.webapp.rendering.offload import RenderRunner
from ptychodus.webapp.rendering.params import InvalidRenderParamError
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
    view_as_png,
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


def _resolve_resource_file(kind: str, uuid: UUID) -> Path:
    """The resource's current file, as its manifest names it; a running one's newest checkpoint."""
    try:
        path = resolve_resource_file(_require_layout(), uuid, kind)
    except (LayoutError, ManifestLoadError) as exc:
        raise ToolError(str(exc)) from exc

    if not path.is_file():
        raise ToolError(f'{path.name} for {kind} {uuid} not present on disk')

    return path


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


async def _ensure_row_exists(kind: str, uuid: UUID) -> None:
    async with _session() as session:
        row = await repo.get_row_by_kind(session, kind, uuid)
        if row is None:
            raise ToolError(f'{kind} {uuid} not found')


def _coerce_render_params_or_error(
    colormap: str,
    transform: str,
    component: str | None,
    color_model: str | None,
    value_min: float | None,
    value_max: float | None,
    clip: bool,
):  # -> RenderParams (typing.TYPE_CHECKING avoided to keep import surface small)
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
        raise ToolError(str(exc)) from exc


def create_mcp_server() -> FastMCP:
    mcp = FastMCP(name='ptychodus-webapp')

    @mcp.tool()
    async def list_campaign(
        limit: int = 50,
        offset: int = 0,
        sample_name: str | None = None,
        ingest_state: str | None = None,
    ) -> Page[CampaignRead]:
        """List campaigns."""
        async with _session() as session:
            where = []
            if sample_name is not None:
                where.append(Campaign.sample_name == sample_name)
            if ingest_state is not None:
                where.append(Campaign.ingest_state == ingest_state)
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
    async def get_campaign(uuid: str) -> CampaignRead | None:
        """Get a campaign by UUID."""
        async with _session() as session:
            row = await repo.get_row(session, Campaign, UUID(uuid))
            return campaign_to_read(row) if row is not None else None

    @mcp.tool()
    async def list_diffraction(
        limit: int = 50,
        offset: int = 0,
        campaign_uuid: str | None = None,
        derived_from_uuid: str | None = None,
        photon_energy_eV_min: float | None = None,  # noqa: N803
        photon_energy_eV_max: float | None = None,  # noqa: N803
        tomography_angle_deg_min: float | None = None,
        tomography_angle_deg_max: float | None = None,
        tilt_angle_deg_min: float | None = None,
        tilt_angle_deg_max: float | None = None,
        polarization: str | None = None,
        ingest_state: str | None = None,
        run_state: list[RunState] | None = None,
    ) -> Page[DiffractionRead]:
        """List diffraction datasets.

        `run_state` keeps only resources whose run is in one of those states (pending, running,
        succeeded, failed, cancelled); imported data has no run.
        """
        async with _session() as session:
            where = []
            if campaign_uuid is not None:
                where.append(Diffraction.campaign_uuid == UUID(campaign_uuid))
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
                try:
                    parsed_pol = Polarization(polarization)
                except ValueError as exc:
                    raise ToolError(
                        f'Unknown polarization {polarization!r}; '
                        f'expected one of {[p.value for p in Polarization]}.'
                    ) from exc
                where.append(Diffraction.polarization == parsed_pol.value)
            if derived_from_uuid is not None:
                target = UUID(derived_from_uuid)
                where.append(
                    Diffraction.uuid.in_(
                        select(DerivationEdge.source_uuid).where(
                            DerivationEdge.target_uuid == target
                        )
                    )
                )
            items, total = await repo.list_rows(
                session, Diffraction, limit=limit, offset=offset, where=where
            )
            reads = [await diffraction_to_read(session, i) for i in items]
            return Page(items=reads, total=total, limit=limit, offset=offset)

    @mcp.tool()
    async def get_diffraction(uuid: str) -> DiffractionRead | None:
        """Get a diffraction dataset by UUID."""
        async with _session() as session:
            row = await repo.get_row(session, Diffraction, UUID(uuid))
            return await diffraction_to_read(session, row) if row is not None else None

    @mcp.tool()
    async def list_product(
        limit: int = 50,
        offset: int = 0,
        derived_from_uuid: str | None = None,
        ingest_state: str | None = None,
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
                target = UUID(derived_from_uuid)
                where.append(
                    Product.uuid.in_(
                        select(DerivationEdge.source_uuid).where(
                            DerivationEdge.target_uuid == target
                        )
                    )
                )
            items, total = await repo.list_rows(
                session, Product, limit=limit, offset=offset, where=where
            )
            reads = [await product_to_read(session, i) for i in items]
            return Page(items=reads, total=total, limit=limit, offset=offset)

    @mcp.tool()
    async def get_product(uuid: str) -> ProductRead | None:
        """Get a product by UUID."""
        async with _session() as session:
            row = await repo.get_row(session, Product, UUID(uuid))
            return await product_to_read(session, row) if row is not None else None

    @mcp.tool()
    async def list_fluorescence(
        limit: int = 50,
        offset: int = 0,
        derived_from_uuid: str | None = None,
        element: str | None = None,
        ingest_state: str | None = None,
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
            if derived_from_uuid is not None:
                target = UUID(derived_from_uuid)
                where.append(
                    Fluorescence.uuid.in_(
                        select(DerivationEdge.source_uuid).where(
                            DerivationEdge.target_uuid == target
                        )
                    )
                )
            if element is not None:
                where.append(
                    func.instr(func.cast(Fluorescence.element_names, String), f'"{element}"') > 0
                )
            items, total = await repo.list_rows(
                session, Fluorescence, limit=limit, offset=offset, where=where
            )
            reads = [await fluorescence_to_read(session, i) for i in items]
            return Page(items=reads, total=total, limit=limit, offset=offset)

    @mcp.tool()
    async def get_fluorescence(uuid: str) -> FluorescenceRead | None:
        """Get a fluorescence dataset by UUID."""
        async with _session() as session:
            row = await repo.get_row(session, Fluorescence, UUID(uuid))
            return await fluorescence_to_read(session, row) if row is not None else None

    @mcp.tool()
    async def get_lineage(uuid: str) -> LineageRead | None:
        """Walk the derivation DAG up and down from a node."""
        from ptychodus.webapp.routers.lineage import (
            _find_campaign,
            _resolve_node,
            _walk_ancestors,
            _walk_descendants,
        )

        target = UUID(uuid)
        async with _session() as session:
            resolved = await _resolve_node(session, target)
            if resolved is None:
                return None
            kind, row = resolved
            node = LineageNode(kind=kind, uuid=target, label=row_label(row))
            ancestors = await _walk_ancestors(session, target)
            descendants = await _walk_descendants(session, target)
            campaign = await _find_campaign(session, target, ancestors)
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

    @mcp.tool()
    async def render_diffraction_pattern(
        uuid: str,
        index: int,
        colormap: str = 'gray',
        transform: str = 'identity',
        value_min: float | None = None,
        value_max: float | None = None,
        clip: bool = False,
    ) -> MCPImage:
        """Render a single diffraction pattern from an assembled dataset."""
        target = UUID(uuid)
        await _ensure_row_exists(ResourceKind.DIFFRACTION, target)
        path = _resolve_resource_file(ResourceKind.DIFFRACTION, target)
        params = _coerce_render_params_or_error(
            colormap, transform, None, None, value_min, value_max, clip
        )
        png = await _render(view_as_png, diffraction_pattern_view, (path, index), params)
        return MCPImage(data=png, format='png')

    @mcp.tool()
    async def render_diffraction_aggregate(
        uuid: str,
        colormap: str = 'gray',
        transform: str = 'identity',
        value_min: float | None = None,
        value_max: float | None = None,
        clip: bool = False,
    ) -> MCPImage:
        """Render the mean pattern across an assembled diffraction dataset."""
        target = UUID(uuid)
        await _ensure_row_exists(ResourceKind.DIFFRACTION, target)
        path = _resolve_resource_file(ResourceKind.DIFFRACTION, target)
        params = _coerce_render_params_or_error(
            colormap, transform, None, None, value_min, value_max, clip
        )
        png = await _render(view_as_png, diffraction_aggregate_view, (path,), params)
        return MCPImage(data=png, format='png')

    @mcp.tool()
    async def render_probe(
        uuid: str,
        incoherent: int = 0,
        colormap: str = 'gray',
        transform: str = 'identity',
        component: str | None = None,
        color_model: str | None = None,
        value_min: float | None = None,
        value_max: float | None = None,
        clip: bool = False,
    ) -> MCPImage:
        """Render a single incoherent probe mode. coherent axis is fixed at 0."""
        target = UUID(uuid)
        await _ensure_row_exists(ResourceKind.PRODUCT, target)
        path = _resolve_resource_file(ResourceKind.PRODUCT, target)
        params = _coerce_render_params_or_error(
            colormap, transform, component, color_model, value_min, value_max, clip
        )
        png = await _render(view_as_png, probe_view, (path, incoherent), params)
        return MCPImage(data=png, format='png')

    @mcp.tool()
    async def render_probe_modes(
        uuid: str,
        colormap: str = 'gray',
        transform: str = 'identity',
        component: str | None = None,
        color_model: str | None = None,
        value_min: float | None = None,
        value_max: float | None = None,
        clip: bool = False,
    ) -> MCPImage:
        """Render all incoherent probe modes tiled horizontally into a single image."""
        target = UUID(uuid)
        await _ensure_row_exists(ResourceKind.PRODUCT, target)
        path = _resolve_resource_file(ResourceKind.PRODUCT, target)
        params = _coerce_render_params_or_error(
            colormap, transform, component, color_model, value_min, value_max, clip
        )
        png = await _render(view_as_png, probe_modes_view, (path,), params)
        return MCPImage(data=png, format='png')

    @mcp.tool()
    async def render_object(
        uuid: str,
        colormap: str = 'gray',
        transform: str = 'identity',
        component: str | None = None,
        color_model: str | None = None,
        value_min: float | None = None,
        value_max: float | None = None,
        clip: bool = False,
    ) -> MCPImage:
        """Render the object as the product of its layers, the transmission of the whole stack."""
        target = UUID(uuid)
        await _ensure_row_exists(ResourceKind.PRODUCT, target)
        path = _resolve_resource_file(ResourceKind.PRODUCT, target)
        params = _coerce_render_params_or_error(
            colormap, transform, component, color_model, value_min, value_max, clip
        )
        png = await _render(view_as_png, object_view, (path,), params)
        return MCPImage(data=png, format='png')

    @mcp.tool()
    async def render_object_layer(
        uuid: str,
        layer: int,
        colormap: str = 'gray',
        transform: str = 'identity',
        component: str | None = None,
        color_model: str | None = None,
        value_min: float | None = None,
        value_max: float | None = None,
        clip: bool = False,
    ) -> MCPImage:
        """Render a single object layer as a complex-valued image."""
        target = UUID(uuid)
        await _ensure_row_exists(ResourceKind.PRODUCT, target)
        path = _resolve_resource_file(ResourceKind.PRODUCT, target)
        params = _coerce_render_params_or_error(
            colormap, transform, component, color_model, value_min, value_max, clip
        )
        png = await _render(view_as_png, object_layer_view, (path, layer), params)
        return MCPImage(data=png, format='png')

    @mcp.tool()
    async def render_scan_positions(
        uuid: list[str],
        connect_path: bool = True,
        width_px: int = 640,
        height_px: int = 640,
    ) -> MCPImage:
        """Plot scan paths, one color and legend entry per product.

        Pass several UUIDs to overlay them on shared axes and compare scans.
        """
        entries: list[tuple[Path, str]] = []

        for raw_uuid in uuid:
            target = UUID(raw_uuid)
            await _ensure_row_exists(ResourceKind.PRODUCT, target)
            entries.append((_resolve_resource_file(ResourceKind.PRODUCT, target), raw_uuid[:8]))

        plot = await _render(
            partial(
                scan_paths_view, connect_path=connect_path, width_px=width_px, height_px=height_px
            ),
            entries,
        )
        return MCPImage(data=b64decode(plot.png_base64), format='png')

    async def _paired_product_file(product_uuid: str | None) -> Path | None:
        if product_uuid is None:
            return None

        product_target = UUID(product_uuid)
        await _ensure_row_exists(ResourceKind.PRODUCT, product_target)
        return _resolve_resource_file(ResourceKind.PRODUCT, product_target)

    @mcp.tool()
    async def render_fluorescence_summary(
        uuid: str,
        product_uuid: str | None = None,
        colormap: str = 'gray',
        transform: str = 'identity',
        value_min: float | None = None,
        value_max: float | None = None,
        clip: bool = False,
    ) -> MCPImage:
        """Render the elementwise sum of every element map in a fluorescence dataset."""
        target = UUID(uuid)
        await _ensure_row_exists(ResourceKind.FLUORESCENCE, target)
        path = _resolve_resource_file(ResourceKind.FLUORESCENCE, target)
        product_path = await _paired_product_file(product_uuid)
        params = _coerce_render_params_or_error(
            colormap, transform, None, None, value_min, value_max, clip
        )
        png = await _render(view_as_png, fluorescence_summary_view, (path, product_path), params)
        return MCPImage(data=png, format='png')

    @mcp.tool()
    async def render_fluorescence_element(
        uuid: str,
        name: str,
        product_uuid: str | None = None,
        colormap: str = 'gray',
        transform: str = 'identity',
        value_min: float | None = None,
        value_max: float | None = None,
        clip: bool = False,
    ) -> MCPImage:
        """Render one element map from a fluorescence dataset."""
        target = UUID(uuid)
        await _ensure_row_exists(ResourceKind.FLUORESCENCE, target)
        path = _resolve_resource_file(ResourceKind.FLUORESCENCE, target)
        product_path = await _paired_product_file(product_uuid)
        params = _coerce_render_params_or_error(
            colormap, transform, None, None, value_min, value_max, clip
        )
        png = await _render(
            view_as_png, fluorescence_element_view, (path, name, product_path), params
        )
        return MCPImage(data=png, format='png')

    mcp.mount(create_xraydb_mcp(), namespace='xraydb')

    return mcp

"""fastmcp server with read-only tools mirroring the REST surface."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.utilities.types import Image as MCPImage
from sqlalchemy import String, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ptychodus.api.diffraction import Polarization
from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.io import load_diffraction_data, load_fluorescence_data, load_product
from ptychodus.api.visualization import (
    ComplexComponent,
    CylindricalColorModel,
    ScalarTransformation,
    cyclic_colormap_names,
    linear_colormap_names,
)

from ptychodus_store.db import repositories as repo
from ptychodus_store.db.base import IngestState
from ptychodus_store.db.models import Campaign, DerivationEdge, Diffraction, Fluorescence, Product
from ptychodus_store.db.session import SessionProvider
from ptychodus_store.rendering import (
    OptionsRead,
    RenderParamsError,
    build_visualization_complex,
    build_visualization_real,
    coerce_render_params,
    product_to_png_bytes,
)
from ptychodus_store.rendering.params import InvalidRenderParamError
from ptychodus_store.routers._convert import (
    campaign_to_read,
    diffraction_to_read,
    fluorescence_to_read,
    product_to_read,
)
from ptychodus_store.routers.schemas import (
    CampaignRead,
    DiffractionRead,
    FluorescenceRead,
    LineageNode,
    LineageRead,
    Page,
    ProductRead,
    StoreStats,
)
from ptychodus_store.storage.layout import StoreLayout
from ptychodus_store.storage.manifest import ResourceKind

logger = logging.getLogger(__name__)

_DEFAULT_FLUORESCENCE_PIXEL_GEOMETRY = PixelGeometry(width_m=1e-6, height_m=1e-6)


class _Ctx:
    provider: SessionProvider | None = None
    layout: StoreLayout | None = None


def bind_session_provider(provider: SessionProvider) -> None:
    _Ctx.provider = provider


def bind_layout(layout: StoreLayout) -> None:
    _Ctx.layout = layout


@asynccontextmanager
async def _session() -> AsyncIterator[AsyncSession]:
    if _Ctx.provider is None:
        raise RuntimeError('SessionProvider not bound; call bind_session_provider() first')
    async with _Ctx.provider.session_factory() as session:
        yield session


def _require_layout() -> StoreLayout:
    if _Ctx.layout is None:
        raise ToolError('StoreLayout not bound; call bind_layout() first')
    return _Ctx.layout


def _resolve_resource_file(kind: str, uuid: UUID, filename: str) -> Path:
    layout = _require_layout()
    path = layout.resource_folder(kind, uuid) / filename
    if not path.is_file():
        raise ToolError(f'{filename} for {kind} {uuid} not present on disk')
    return path


async def _ensure_row_exists(kind: str, uuid: UUID) -> None:
    async with _session() as session:
        row = await repo.get_row(session, kind, uuid)
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
    mcp = FastMCP(name='ptychodus-store')

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
                session, ResourceKind.CAMPAIGN, limit=limit, offset=offset, where=where
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
            row = await repo.get_row(session, ResourceKind.CAMPAIGN, UUID(uuid))
            return campaign_to_read(row) if row is not None else None

    @mcp.tool()
    async def list_diffraction(
        limit: int = 50,
        offset: int = 0,
        campaign_uuid: str | None = None,
        derived_from_uuid: str | None = None,
        probe_energy_eV_min: float | None = None,  # noqa: N803
        probe_energy_eV_max: float | None = None,  # noqa: N803
        tilt_angle_deg_min: float | None = None,
        tilt_angle_deg_max: float | None = None,
        polarization: str | None = None,
        ingest_state: str | None = None,
    ) -> Page[DiffractionRead]:
        """List diffraction datasets."""
        async with _session() as session:
            where = []
            if campaign_uuid is not None:
                where.append(Diffraction.campaign_uuid == UUID(campaign_uuid))
            if ingest_state is not None:
                where.append(Diffraction.ingest_state == ingest_state)
            if probe_energy_eV_min is not None:
                where.append(Diffraction.probe_energy_eV >= probe_energy_eV_min)
            if probe_energy_eV_max is not None:
                where.append(Diffraction.probe_energy_eV <= probe_energy_eV_max)
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
                session, ResourceKind.DIFFRACTION, limit=limit, offset=offset, where=where
            )
            reads = [await diffraction_to_read(session, i) for i in items]  # type: ignore[arg-type]
            return Page(items=reads, total=total, limit=limit, offset=offset)

    @mcp.tool()
    async def get_diffraction(uuid: str) -> DiffractionRead | None:
        """Get a diffraction dataset by UUID."""
        async with _session() as session:
            row = await repo.get_row(session, ResourceKind.DIFFRACTION, UUID(uuid))
            return await diffraction_to_read(session, row) if row is not None else None  # type: ignore[arg-type]

    @mcp.tool()
    async def list_product(
        limit: int = 50,
        offset: int = 0,
        derived_from_uuid: str | None = None,
        ingest_state: str | None = None,
    ) -> Page[ProductRead]:
        """List reconstruction products."""
        async with _session() as session:
            where = []
            if ingest_state is not None:
                where.append(Product.ingest_state == ingest_state)
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
                session, ResourceKind.PRODUCT, limit=limit, offset=offset, where=where
            )
            reads = [await product_to_read(session, i) for i in items]  # type: ignore[arg-type]
            return Page(items=reads, total=total, limit=limit, offset=offset)

    @mcp.tool()
    async def get_product(uuid: str) -> ProductRead | None:
        """Get a product by UUID."""
        async with _session() as session:
            row = await repo.get_row(session, ResourceKind.PRODUCT, UUID(uuid))
            return await product_to_read(session, row) if row is not None else None  # type: ignore[arg-type]

    @mcp.tool()
    async def list_fluorescence(
        limit: int = 50,
        offset: int = 0,
        derived_from_uuid: str | None = None,
        element: str | None = None,
    ) -> Page[FluorescenceRead]:
        """List fluorescence datasets."""
        async with _session() as session:
            where = []
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
                    func.instr(func.cast(Fluorescence.element_names, String), f'"{element}"') > 0  # type: ignore[arg-type]
                )
            items, total = await repo.list_rows(
                session, ResourceKind.FLUORESCENCE, limit=limit, offset=offset, where=where
            )
            reads = [await fluorescence_to_read(session, i) for i in items]  # type: ignore[arg-type]
            return Page(items=reads, total=total, limit=limit, offset=offset)

    @mcp.tool()
    async def get_fluorescence(uuid: str) -> FluorescenceRead | None:
        """Get a fluorescence dataset by UUID."""
        async with _session() as session:
            row = await repo.get_row(session, ResourceKind.FLUORESCENCE, UUID(uuid))
            return await fluorescence_to_read(session, row) if row is not None else None  # type: ignore[arg-type]

    @mcp.tool()
    async def get_lineage(uuid: str) -> LineageRead | None:
        """Walk the derivation DAG up and down from a node."""
        from ptychodus_store.routers.lineage import (
            _find_campaign,
            _label_for,
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
            node = LineageNode(kind=kind, uuid=target, label=_label_for(row))  # type: ignore[arg-type]
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

            async def _count(model):  # type: ignore[no-untyped-def]
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
        return OptionsRead(
            colormaps_linear=list(linear_colormap_names()),
            colormaps_cyclic=list(cyclic_colormap_names()),
            transforms=[member.name.lower() for member in ScalarTransformation],
            components=[member.name.lower() for member in ComplexComponent],
            color_models=[member.name.lower() for member in CylindricalColorModel],
        )

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
        path = _resolve_resource_file(ResourceKind.DIFFRACTION, target, 'diffraction.h5')
        data = load_diffraction_data(path)
        num_patterns = data.get_patterns_shape()[0]
        if not 0 <= index < num_patterns:
            raise ToolError(f'pattern index {index} out of range [0, {num_patterns})')
        params = _coerce_render_params_or_error(
            colormap, transform, None, None, value_min, value_max, clip
        )
        try:
            vp = build_visualization_real(
                data.get_pattern(index), data.get_pixel_geometry(), params, value_label='Counts'
            )
        except RenderParamsError as exc:
            raise ToolError(str(exc)) from exc
        return MCPImage(data=product_to_png_bytes(vp), format='png')

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
        path = _resolve_resource_file(ResourceKind.DIFFRACTION, target, 'diffraction.h5')
        data = load_diffraction_data(path)
        params = _coerce_render_params_or_error(
            colormap, transform, None, None, value_min, value_max, clip
        )
        try:
            vp = build_visualization_real(
                data.get_average_pattern(),
                data.get_pixel_geometry(),
                params,
                value_label='Mean Counts',
            )
        except RenderParamsError as exc:
            raise ToolError(str(exc)) from exc
        return MCPImage(data=product_to_png_bytes(vp), format='png')

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
        path = _resolve_resource_file(ResourceKind.PRODUCT, target, 'product.h5')
        product = load_product(path)
        probe = product.probes.get_probe_no_opr()
        if not 0 <= incoherent < probe.num_incoherent_modes:
            raise ToolError(
                f'incoherent mode {incoherent} out of range [0, {probe.num_incoherent_modes})'
            )
        params = _coerce_render_params_or_error(
            colormap, transform, component, color_model, value_min, value_max, clip
        )
        try:
            vp = build_visualization_complex(
                probe.get_incoherent_mode(incoherent),
                product.probes.get_pixel_geometry(),
                params,
            )
        except RenderParamsError as exc:
            raise ToolError(str(exc)) from exc
        return MCPImage(data=product_to_png_bytes(vp), format='png')

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
        path = _resolve_resource_file(ResourceKind.PRODUCT, target, 'product.h5')
        product = load_product(path)
        probe = product.probes.get_probe_no_opr()
        params = _coerce_render_params_or_error(
            colormap, transform, component, color_model, value_min, value_max, clip
        )
        try:
            vp = build_visualization_complex(
                probe.get_incoherent_modes_flattened(),
                product.probes.get_pixel_geometry(),
                params,
            )
        except RenderParamsError as exc:
            raise ToolError(str(exc)) from exc
        return MCPImage(data=product_to_png_bytes(vp), format='png')

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
        path = _resolve_resource_file(ResourceKind.PRODUCT, target, 'product.h5')
        product = load_product(path)
        if not 0 <= layer < product.object_.num_layers:
            raise ToolError(f'object layer {layer} out of range [0, {product.object_.num_layers})')
        params = _coerce_render_params_or_error(
            colormap, transform, component, color_model, value_min, value_max, clip
        )
        try:
            vp = build_visualization_complex(
                product.object_.get_layer(layer),
                product.object_.get_pixel_geometry(),
                params,
            )
        except RenderParamsError as exc:
            raise ToolError(str(exc)) from exc
        return MCPImage(data=product_to_png_bytes(vp), format='png')

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
        path = _resolve_resource_file(ResourceKind.FLUORESCENCE, target, 'fluorescence.h5')
        dataset = load_fluorescence_data(path)
        matches = [emap for emap in dataset.element_maps if emap.name == name]
        if not matches:
            available = ', '.join(emap.name for emap in dataset.element_maps)
            raise ToolError(f'element {name!r} not found; available: [{available}]')
        emap = matches[0]

        if product_uuid is not None:
            product_target = UUID(product_uuid)
            await _ensure_row_exists(ResourceKind.PRODUCT, product_target)
            product_path = _resolve_resource_file(
                ResourceKind.PRODUCT, product_target, 'product.h5'
            )
            pixel_geometry = load_product(product_path).object_.get_pixel_geometry()
        else:
            pixel_geometry = _DEFAULT_FLUORESCENCE_PIXEL_GEOMETRY

        params = _coerce_render_params_or_error(
            colormap, transform, None, None, value_min, value_max, clip
        )
        try:
            vp = build_visualization_real(
                emap.counts_per_second,
                pixel_geometry,
                params,
                value_label=f'{emap.name} counts/s',
            )
        except RenderParamsError as exc:
            raise ToolError(str(exc)) from exc
        return MCPImage(data=product_to_png_bytes(vp), format='png')

    try:
        from ptychodus_store.mcp_tools.xraydb import create_xraydb_mcp

        mcp.mount(create_xraydb_mcp(), namespace='xraydb')
        logger.info('mounted xraydb MCP sub-server (tools namespaced as xraydb_*)')
    except ImportError:
        logger.debug('xraydb not installed; xraydb MCP tools disabled')

    return mcp

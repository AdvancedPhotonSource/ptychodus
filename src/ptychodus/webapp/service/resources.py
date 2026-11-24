"""The read operations on the indexed resources, each declaring its operation class.

Every operation returns a pydantic read model or raises
:class:`~ptychodus.webapp.service.errors.ResourceNotFoundError` or
:class:`~ptychodus.webapp.service.errors.InvalidRequestError`, whatever transport carries it.
An instant operation reads the metadata cache. A bounded one looks its resources up there,
then reads and computes on their files through the shared
:class:`~ptychodus.webapp.rendering.offload.RenderRunner`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy import String, exists, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ptychodus.api.diffraction import Polarization
from ptychodus.api.metrics import SpectralSNRConvention
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
from ptychodus.webapp.rendering.offload import RenderRunner
from ptychodus.webapp.rendering.params import RenderParams, visualization_options
from ptychodus.webapp.rendering.schemas import OptionsRead, PlotImage, RenderedImage
from ptychodus.webapp.rendering.views import (
    diffraction_aggregate_view,
    diffraction_pattern_view,
    fluorescence_element_view,
    fluorescence_summary_view,
    object_layer_view,
    object_view,
    probe_modes_view,
    probe_positions_view,
    probe_view,
    view_as_rendered_image,
)
from ptychodus.webapp.service.analysis import (
    object_comparison_view,
    object_metrics_view,
    probe_comparison_view,
    probe_metrics_view,
    probe_positions_comparison_view,
    probe_positions_metrics_view,
)
from ptychodus.webapp.service.convert import (
    campaign_to_read,
    diffraction_to_read,
    fluorescence_to_read,
    product_to_read,
    row_label,
    run_to_read,
)
from ptychodus.webapp.service.errors import ResourceNotFoundError
from ptychodus.webapp.service.lineage import (
    find_campaign,
    resolve_node,
    walk_ancestors,
    walk_descendants,
)
from ptychodus.webapp.service.operations import OperationClass, operation
from ptychodus.webapp.service.schemas import (
    CampaignRead,
    DiffractionRead,
    FluorescenceRead,
    LineageNode,
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
    RunRead,
    SpectralSNRConventionName,
)

__all__ = ['ResourceService']

_INSTANT = operation(OperationClass.INSTANT)
_BOUNDED = operation(OperationClass.BOUNDED)


def _json_array_contains(column: Any, value: str) -> Any:
    """Whether a JSON-encoded TEXT array column holds `value`, by its quoted JSON form."""
    return func.instr(func.cast(column, String), f'"{value}"') > 0


def _derived_from(model: type[Diffraction | Product | Fluorescence], target: UUID) -> Any:
    """Whether a row of `model` names `target` among its `derived_from` edges."""
    edge = select(DerivationEdge.source_uuid).where(
        DerivationEdge.source_uuid == model.uuid,
        DerivationEdge.target_uuid == target,
    )
    return exists(edge)


class ResourceService:
    """The webapp's read operations, over one metadata cache and one storage root."""

    def __init__(
        self, session_provider: SessionProvider, layout: StorageLayout, runner: RenderRunner
    ) -> None:
        self._session_provider = session_provider
        self._layout = layout
        self._runner = runner

    @asynccontextmanager
    async def _session(self) -> AsyncIterator[AsyncSession]:
        async with self._session_provider.session_factory() as session:
            yield session

    def _current_file(self, kind: str, uuid: UUID) -> Path:
        """The file holding an indexed resource's current data, as its manifest names it.

        A running resource names its newest checkpoint, and one with no result yet has none.
        """
        try:
            path = resolve_resource_file(self._layout, uuid, kind)
        except (LayoutError, ManifestLoadError) as exc:
            raise ResourceNotFoundError(str(exc)) from exc

        if not path.is_file():
            raise ResourceNotFoundError(f'{path.name} not present on disk')

        return path

    async def _product_file_and_run(self, uuid: UUID) -> tuple[Path, RunRead | None]:
        async with self._session() as session:
            row = await repo.get_row(session, Product, uuid)

        if row is None:
            raise ResourceNotFoundError(f'product {uuid} not found')

        return self._current_file(ResourceKind.PRODUCT, uuid), run_to_read(row)

    async def _paired_product_file(self, product_uuid: UUID | None) -> Path | None:
        """The paired product whose object pixel geometry a map takes, if one was named."""
        if product_uuid is None:
            return None

        async with self._session() as session:
            row = await repo.get_row(session, Product, product_uuid)

        if row is None:
            raise ResourceNotFoundError(f'product {product_uuid} not found for pixel geometry')

        return self._current_file(ResourceKind.PRODUCT, product_uuid)

    # Instant operations
    # ----------------------------------------------------------------------------------

    @_INSTANT
    async def list_campaign(
        self,
        *,
        limit: int,
        offset: int,
        sample_name: str | None = None,
        tag: str | None = None,
        ingest_state: IngestState | None = None,
    ) -> Page[CampaignRead]:
        """List campaigns, optionally only those carrying `tag`."""
        where = []
        if sample_name is not None:
            where.append(Campaign.sample_name == sample_name)
        if ingest_state is not None:
            where.append(Campaign.ingest_state == ingest_state)
        if tag is not None:
            where.append(_json_array_contains(Campaign.tags, tag))

        async with self._session() as session:
            items, total = await repo.list_rows(
                session, Campaign, limit=limit, offset=offset, where=where
            )
            reads = [campaign_to_read(i) for i in items]

        return Page(items=reads, total=total, limit=limit, offset=offset)

    @_INSTANT
    async def get_campaign(self, uuid: UUID) -> CampaignRead:
        """Get a campaign by uuid."""
        async with self._session() as session:
            row = await repo.get_row(session, Campaign, uuid)

        if row is None:
            raise ResourceNotFoundError(f'campaign {uuid} not found')

        return campaign_to_read(row)

    @_INSTANT
    async def list_diffraction(
        self,
        *,
        limit: int,
        offset: int,
        campaign_uuid: UUID | None = None,
        derived_from_uuid: UUID | None = None,
        ingest_state: IngestState | None = None,
        run_state: Sequence[RunState] | None = None,
        photon_energy_eV_min: float | None = None,  # noqa: N803
        photon_energy_eV_max: float | None = None,  # noqa: N803
        tomography_angle_deg_min: float | None = None,
        tomography_angle_deg_max: float | None = None,
        tilt_angle_deg_min: float | None = None,
        tilt_angle_deg_max: float | None = None,
        polarization: Polarization | None = None,
    ) -> Page[DiffractionRead]:
        """List diffraction datasets.

        `run_state` keeps only resources whose run is in one of those states; imported data
        has no run.
        """
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
            where.append(_derived_from(Diffraction, derived_from_uuid))

        async with self._session() as session:
            items, total = await repo.list_rows(
                session, Diffraction, limit=limit, offset=offset, where=where
            )
            reads = [await diffraction_to_read(session, i) for i in items]

        return Page(items=reads, total=total, limit=limit, offset=offset)

    @_INSTANT
    async def get_diffraction(self, uuid: UUID) -> DiffractionRead:
        """Get a diffraction dataset by uuid."""
        async with self._session() as session:
            row = await repo.get_row(session, Diffraction, uuid)

            if row is None:
                raise ResourceNotFoundError(f'diffraction {uuid} not found')

            return await diffraction_to_read(session, row)

    @_INSTANT
    async def list_product(
        self,
        *,
        limit: int,
        offset: int,
        derived_from_uuid: UUID | None = None,
        ingest_state: IngestState | None = None,
        run_state: Sequence[RunState] | None = None,
        photon_energy_eV_min: float | None = None,  # noqa: N803
        photon_energy_eV_max: float | None = None,  # noqa: N803
        scattering_regime: ScatteringRegime | None = None,
    ) -> Page[ProductRead]:
        """List products, optionally only transmission, Bragg or Laue ones.

        `run_state` keeps only resources whose run is in one of those states; imported data
        has no run.
        """
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
            where.append(_derived_from(Product, derived_from_uuid))

        async with self._session() as session:
            items, total = await repo.list_rows(
                session, Product, limit=limit, offset=offset, where=where
            )
            reads = [await product_to_read(session, i) for i in items]

        return Page(items=reads, total=total, limit=limit, offset=offset)

    @_INSTANT
    async def get_product(self, uuid: UUID) -> ProductRead:
        """Get a product by uuid."""
        async with self._session() as session:
            row = await repo.get_row(session, Product, uuid)

            if row is None:
                raise ResourceNotFoundError(f'product {uuid} not found')

            return await product_to_read(session, row)

    @_INSTANT
    async def list_fluorescence(
        self,
        *,
        limit: int,
        offset: int,
        derived_from_uuid: UUID | None = None,
        element: str | None = None,
        ingest_state: IngestState | None = None,
        run_state: Sequence[RunState] | None = None,
    ) -> Page[FluorescenceRead]:
        """List fluorescence datasets, optionally only those with an `element` map.

        `run_state` keeps only resources whose run is in one of those states; imported data
        has no run.
        """
        where = []
        if ingest_state is not None:
            where.append(Fluorescence.ingest_state == ingest_state)
        if run_state:
            where.append(Fluorescence.run_state.in_([s.value for s in run_state]))
        if element is not None:
            where.append(_json_array_contains(Fluorescence.element_names, element))
        if derived_from_uuid is not None:
            where.append(_derived_from(Fluorescence, derived_from_uuid))

        async with self._session() as session:
            items, total = await repo.list_rows(
                session, Fluorescence, limit=limit, offset=offset, where=where
            )
            reads = [await fluorescence_to_read(session, i) for i in items]

        return Page(items=reads, total=total, limit=limit, offset=offset)

    @_INSTANT
    async def get_fluorescence(self, uuid: UUID) -> FluorescenceRead:
        """Get a fluorescence dataset by uuid."""
        async with self._session() as session:
            row = await repo.get_row(session, Fluorescence, uuid)

            if row is None:
                raise ResourceNotFoundError(f'fluorescence {uuid} not found')

            return await fluorescence_to_read(session, row)

    @_INSTANT
    async def get_lineage(self, uuid: UUID) -> LineageRead:
        """Walk the derivation DAG up and down from a resource."""
        async with self._session() as session:
            resolved = await resolve_node(session, uuid)

            if resolved is None:
                raise ResourceNotFoundError(f'no resource with uuid {uuid}')

            kind, row = resolved
            node = LineageNode(kind=kind, uuid=uuid, label=row_label(row))
            ancestors = await walk_ancestors(session, uuid)
            descendants = await walk_descendants(session, uuid)
            campaign = await find_campaign(session, uuid, ancestors)

        return LineageRead(
            node=node, ancestors=ancestors, descendants=descendants, campaign=campaign
        )

    @_INSTANT
    async def get_resource_stats(self) -> ResourceStats:
        """Count the resources the webapp has indexed, and the invalid ones."""
        async with self._session() as session:
            return ResourceStats(
                campaign_count=await repo.count_rows(session, Campaign),
                diffraction_count=await repo.count_rows(session, Diffraction),
                product_count=await repo.count_rows(session, Product),
                fluorescence_count=await repo.count_rows(session, Fluorescence),
                invalid_count=await repo.count_invalid_rows(session),
            )

    @_INSTANT
    async def get_visualization_options(self) -> OptionsRead:
        """Enumerate the valid choices for colormap, transform, component and color_model."""
        return visualization_options()

    @_INSTANT
    async def resource_file(self, kind: str, uuid: UUID) -> Path:
        """The file holding a resource's current data: its final file, or newest checkpoint."""
        async with self._session() as session:
            row = await repo.get_row_by_kind(session, kind, uuid)

        if row is None:
            raise ResourceNotFoundError(f'{kind} {uuid} not found')

        return self._current_file(kind, uuid)

    # Bounded operations: renders
    # ----------------------------------------------------------------------------------

    @_BOUNDED
    async def render_diffraction_pattern(
        self, uuid: UUID, index: int, params: RenderParams
    ) -> RenderedImage:
        """Render a single diffraction pattern from an assembled dataset, with its scales."""
        path = await self.resource_file(ResourceKind.DIFFRACTION, uuid)
        return await self._runner.run(
            view_as_rendered_image, diffraction_pattern_view, (path, index), params, False
        )

    @_BOUNDED
    async def render_diffraction_aggregate(self, uuid: UUID, params: RenderParams) -> RenderedImage:
        """Render the mean pattern across an assembled diffraction dataset, with its scales."""
        path = await self.resource_file(ResourceKind.DIFFRACTION, uuid)
        return await self._runner.run(
            view_as_rendered_image, diffraction_aggregate_view, (path,), params, False
        )

    @_BOUNDED
    async def render_probe(
        self, uuid: UUID, incoherent: int, params: RenderParams
    ) -> RenderedImage:
        """Render a single incoherent probe mode, with its scales. The coherent axis is 0."""
        path = await self.resource_file(ResourceKind.PRODUCT, uuid)
        return await self._runner.run(
            view_as_rendered_image, probe_view, (path, incoherent), params, True
        )

    @_BOUNDED
    async def render_probe_modes(self, uuid: UUID, params: RenderParams) -> RenderedImage:
        """Render all incoherent probe modes tiled horizontally, with their scales."""
        path = await self.resource_file(ResourceKind.PRODUCT, uuid)
        return await self._runner.run(
            view_as_rendered_image, probe_modes_view, (path,), params, True
        )

    @_BOUNDED
    async def render_object(self, uuid: UUID, params: RenderParams) -> RenderedImage:
        """Render the object as the product of its layers, with its scales."""
        path = await self.resource_file(ResourceKind.PRODUCT, uuid)
        return await self._runner.run(view_as_rendered_image, object_view, (path,), params, True)

    @_BOUNDED
    async def render_object_layer(
        self, uuid: UUID, layer: int, params: RenderParams
    ) -> RenderedImage:
        """Render a single object layer as a complex-valued image, with its scales."""
        path = await self.resource_file(ResourceKind.PRODUCT, uuid)
        return await self._runner.run(
            view_as_rendered_image, object_layer_view, (path, layer), params, True
        )

    @_BOUNDED
    async def render_probe_positions(
        self, uuids: Sequence[UUID], *, connect_path: bool, width_px: int, height_px: int
    ) -> PlotImage:
        """Plot probe positions, one color and legend entry per product, with the plot's size.

        Pass several products to overlay them on shared axes and compare them.
        """
        entries = [
            (await self.resource_file(ResourceKind.PRODUCT, uuid), str(uuid)[:8]) for uuid in uuids
        ]
        return await self._runner.run(
            partial(
                probe_positions_view,
                connect_path=connect_path,
                width_px=width_px,
                height_px=height_px,
            ),
            entries,
        )

    @_BOUNDED
    async def render_fluorescence_summary(
        self, uuid: UUID, product_uuid: UUID | None, params: RenderParams
    ) -> RenderedImage:
        """Render the elementwise sum of every element map in a dataset, with its scales."""
        path = await self.resource_file(ResourceKind.FLUORESCENCE, uuid)
        product_path = await self._paired_product_file(product_uuid)
        return await self._runner.run(
            view_as_rendered_image,
            fluorescence_summary_view,
            (path, product_path),
            params,
            False,
        )

    @_BOUNDED
    async def render_fluorescence_element(
        self, uuid: UUID, name: str, product_uuid: UUID | None, params: RenderParams
    ) -> RenderedImage:
        """Render one element map from a fluorescence dataset, with its scales."""
        path = await self.resource_file(ResourceKind.FLUORESCENCE, uuid)
        product_path = await self._paired_product_file(product_uuid)
        return await self._runner.run(
            view_as_rendered_image,
            fluorescence_element_view,
            (path, name, product_path),
            params,
            False,
        )

    # Bounded operations: analysis
    # ----------------------------------------------------------------------------------

    @_BOUNDED
    async def get_probe_metrics(self, uuid: UUID) -> ProbeMetricsRead:
        """Describe a product's probe: its size, entropy, mode spectrum and scan overlap."""
        path, run = await self._product_file_and_run(uuid)
        scalars = await self._runner.run(probe_metrics_view, path)
        return ProbeMetricsRead(uuid=uuid, run=run, scalars=scalars)

    @_BOUNDED
    async def get_object_metrics(self, uuid: UUID, *, erosion_divisor: int) -> ObjectMetricsRead:
        """Describe a product's object over the region its scan illuminated.

        The phase offset and ramp are removed first, so the statistics describe the sample
        rather than an arbitrary tilt.
        """
        path, run = await self._product_file_and_run(uuid)
        scalars = await self._runner.run(object_metrics_view, path, erosion_divisor)
        return ObjectMetricsRead(uuid=uuid, run=run, scalars=scalars)

    @_BOUNDED
    async def get_probe_positions_metrics(self, uuid: UUID) -> ProbePositionsMetricsRead:
        """Describe a product's probe positions: their extent, spacing, scan indexes and I0."""
        path, run = await self._product_file_and_run(uuid)
        scalars = await self._runner.run(probe_positions_metrics_view, path)
        return ProbePositionsMetricsRead(uuid=uuid, run=run, scalars=scalars)

    @_BOUNDED
    async def compare_probe(
        self,
        uuid: UUID,
        reference_uuid: UUID,
        *,
        erosion_divisor: int,
        mad_threshold: float,
        resample_test: bool,
    ) -> ProbeComparisonRead:
        """Compare a product's probe with a reference product's."""
        path, run = await self._product_file_and_run(uuid)
        reference_path, reference_run = await self._product_file_and_run(reference_uuid)
        scalars = await self._runner.run(
            probe_comparison_view,
            reference_path,
            path,
            erosion_divisor,
            mad_threshold,
            resample_test,
        )
        return ProbeComparisonRead(
            uuid=uuid,
            run=run,
            reference_uuid=reference_uuid,
            reference_run=reference_run,
            scalars=scalars,
        )

    @_BOUNDED
    async def compare_object(
        self,
        uuid: UUID,
        reference_uuid: UUID,
        *,
        convention: SpectralSNRConventionName,
        snr_threshold: float,
        erosion_divisor: int,
        resample_test: bool,
    ) -> ObjectComparisonRead:
        """Compare a product's projected object with a reference product's.

        The resolution comes from their Fourier ring correlation, and the spectral SNR
        follows `convention`: `half_split` for reconstructions from complementary halves of
        one dataset, `repeat` for independent repeat scans.
        """
        path, run = await self._product_file_and_run(uuid)
        reference_path, reference_run = await self._product_file_and_run(reference_uuid)
        scalars = await self._runner.run(
            object_comparison_view,
            reference_path,
            path,
            SpectralSNRConvention[convention.upper()],
            snr_threshold,
            erosion_divisor,
            resample_test,
        )
        return ObjectComparisonRead(
            uuid=uuid,
            run=run,
            reference_uuid=reference_uuid,
            reference_run=reference_run,
            scalars=scalars,
        )

    @_BOUNDED
    async def compare_probe_positions(
        self, uuid: UUID, reference_uuid: UUID
    ) -> ProbePositionsComparisonRead:
        """Measure how a product's probe positions are displaced from a reference product's.

        Positions pair at the scan indexes both products have.
        """
        path, run = await self._product_file_and_run(uuid)
        reference_path, reference_run = await self._product_file_and_run(reference_uuid)
        scalars = await self._runner.run(probe_positions_comparison_view, reference_path, path)
        return ProbePositionsComparisonRead(
            uuid=uuid,
            run=run,
            reference_uuid=reference_uuid,
            reference_run=reference_run,
            scalars=scalars,
        )

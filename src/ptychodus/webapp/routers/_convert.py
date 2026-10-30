"""Helpers to convert ORM rows + edges into API read models."""

from __future__ import annotations

from uuid import UUID
import math

from sqlalchemy.ext.asyncio import AsyncSession

from ptychodus.webapp.db import repositories as repo
from ptychodus.api.constants import energy_eV_to_wavelength_m
from ptychodus.api.geometry import ImageExtent, PixelGeometry
from ptychodus.api.propagate import compute_product_geometry

from ptychodus.webapp.db.models import Diffraction, Fluorescence, Product
from ptychodus.webapp.routers.schemas import (
    CampaignRead,
    DerivedFromEdge,
    DiffractionRead,
    FluorescenceRead,
    ProductRead,
)


def row_label(row: object) -> str:
    """Display name of a row: a diffraction dataset's label, a product's name."""
    return str(getattr(row, 'label', '') or getattr(row, 'name', '') or '')


async def _edges_for(session: AsyncSession, uuid: UUID) -> list[DerivedFromEdge]:
    edges = await repo.outgoing_edges(session, uuid)
    out: list[DerivedFromEdge] = []

    for edge in edges:
        target = await repo.get_row_by_kind(session, edge.target_kind, edge.target_uuid)
        label = row_label(target) if target is not None else ''
        out.append(DerivedFromEdge(kind=edge.target_kind, uuid=edge.target_uuid, label=label))

    return out


async def diffraction_to_read(session: AsyncSession, row: Diffraction) -> DiffractionRead:
    data = DiffractionRead.model_validate(row)
    data.derived_from = await _edges_for(session, row.uuid)
    return data


async def _detector_geometry_for(
    session: AsyncSession, edges: list[DerivedFromEdge]
) -> tuple[ImageExtent, PixelGeometry] | None:
    """Detector sampling of the diffraction dataset a product derives from.

    ``product.h5`` records the object-plane pitch but never the detector's, so the
    three derived quantities that need one are only available through the lineage
    edge. Returns None when the manifest declares no diffraction source, or when the
    row it names is absent or never recorded its pixel size -- those products report
    the affected fields as null rather than having a value invented for them.
    """
    for edge in edges:
        if edge.kind != 'diffraction':
            continue

        source = await repo.get_row(session, Diffraction, edge.uuid)

        if source is None:
            continue

        width_m = source.detector_pixel_width_m
        height_m = source.detector_pixel_height_m
        width_px = source.pattern_width_px
        height_px = source.pattern_height_px

        if None in (width_m, height_m, width_px, height_px):
            continue

        return (
            ImageExtent(width_px=int(width_px), height_px=int(height_px)),  # type: ignore[arg-type]
            PixelGeometry(width_m=float(width_m), height_m=float(height_m)),  # type: ignore[arg-type]
        )

    return None


def _finite_or_none(value: float) -> float | None:
    """JSON has no infinity, and an undetermined quantity reads better as null."""
    return value if math.isfinite(value) else None


async def product_to_read(session: AsyncSession, row: Product) -> ProductRead:
    data = ProductRead.model_validate(row)
    data.derived_from = await _edges_for(session, row.uuid)

    if row.object_layers is not None:
        # N layers are separated by N-1 spacings, so a single layer is zero thick.
        data.object_total_thickness_m = sum(row.object_layer_spacing_m)

    if row.photon_energy_eV is not None and row.photon_energy_eV > 0.0:
        data.photon_wavelength_m = energy_eV_to_wavelength_m(row.photon_energy_eV)

    if row.photon_energy_eV is None or row.detector_distance_m is None:
        # Without these nothing downstream is meaningful; leave the derived block null.
        return data

    detector = await _detector_geometry_for(session, data.derived_from)
    extent, pixel_geometry = detector if detector is not None else (None, None)

    geometry = compute_product_geometry(
        photon_energy_eV=row.photon_energy_eV,
        probe_photon_count=row.probe_photon_count or 0.0,
        exposure_time_s=row.exposure_time_s or 0.0,
        detector_distance_m=row.detector_distance_m,
        focus_object_distance_m=row.focus_object_distance_m or 0.0,
        far_field=True if row.far_field is None else row.far_field,
        detector_extent=extent,
        detector_pixel_geometry=pixel_geometry,
    )

    data.probe_wavenumber_per_m = _finite_or_none(geometry.probe_wavenumber_per_m)
    data.probe_angular_wavenumber_rad_per_m = _finite_or_none(
        geometry.probe_angular_wavenumber_rad_per_m
    )
    data.probe_photon_flux_per_s = _finite_or_none(geometry.probe_photon_flux_per_s)
    data.probe_power_W = _finite_or_none(geometry.probe_power_W)
    data.object_plane_propagation_distance_m = _finite_or_none(
        geometry.object_plane_propagation_distance_m
    )
    data.magnification = _finite_or_none(geometry.magnification)

    if detector is not None:
        object_plane_pixel_geometry = geometry.object_plane_pixel_geometry

        # A degenerate geometry reports a zero pitch rather than a divergent one.
        if object_plane_pixel_geometry.width_m > 0.0 and object_plane_pixel_geometry.height_m > 0.0:
            data.object_plane_pixel_width_m = object_plane_pixel_geometry.width_m
            data.object_plane_pixel_height_m = object_plane_pixel_geometry.height_m

        data.fresnel_number = _finite_or_none(geometry.fresnel_number)
        data.detector_numerical_aperture = _finite_or_none(geometry.detector_numerical_aperture)
        data.depth_of_field_m = _finite_or_none(geometry.depth_of_field_m)

    return data


async def fluorescence_to_read(session: AsyncSession, row: Fluorescence) -> FluorescenceRead:
    data = FluorescenceRead.model_validate(row)
    data.derived_from = await _edges_for(session, row.uuid)
    return data


def campaign_to_read(row: object) -> CampaignRead:
    return CampaignRead.model_validate(row)

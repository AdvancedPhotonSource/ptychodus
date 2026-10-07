"""Helpers to convert ORM rows + edges into API read models."""

from __future__ import annotations

from uuid import UUID
import math

from sqlalchemy.ext.asyncio import AsyncSession

from ptychodus_store.db import repositories as repo
from ptychodus.api.geometry import ImageExtent, PixelGeometry
from ptychodus.api.propagate import compute_product_geometry

from ptychodus_store.db.models import Diffraction, Fluorescence, Product
from ptychodus_store.routers.schemas import (
    CampaignRead,
    DerivedFromEdge,
    DiffractionRead,
    FluorescenceRead,
    ProductRead,
)


async def _edges_for(session: AsyncSession, uuid: UUID) -> list[DerivedFromEdge]:
    edges = await repo.outgoing_edges(session, uuid)
    return [DerivedFromEdge(kind=e.target_kind, uuid=e.target_uuid) for e in edges]  # type: ignore[arg-type]


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

    if detector is not None:
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

"""Illumination-map data structures and the algorithm that builds a photon-count canvas
from a ptychography product by summing subpixel-shifted probe intensities."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import numpy

from .common import RealArrayType
from .geometry import PixelGeometry, fourier_shift_2d
from .object import ObjectCenter
from .product import Product


@dataclass(frozen=True)
class IlluminationMap:
    """Per-object-pixel photon count plus the metadata needed to derive fluence,
    dose, and intensity quantities."""

    photon_number: RealArrayType
    photon_flux_Hz: float  # noqa: N815
    photon_energy_J: float  # noqa: N815
    exposure_time_s: float
    mass_attenuation_m2_kg: float
    pixel_geometry: PixelGeometry
    center: ObjectCenter

    @property
    def photon_fluence_1_m2(self) -> RealArrayType:
        return self.photon_number / self.pixel_geometry.get_area_m2()

    @property
    def photon_fluence_rate_Hz_m2(self) -> RealArrayType:  # noqa: N802
        return self.photon_fluence_1_m2 / self.exposure_time_s

    @property
    def energy_fluence_J_m2(self) -> RealArrayType:  # noqa: N802
        return self.photon_fluence_1_m2 * self.photon_energy_J

    @property
    def energy_fluence_rate_W_m2(self) -> RealArrayType:  # noqa: N802
        return self.photon_fluence_rate_Hz_m2 * self.photon_energy_J

    @property
    def dose_Gy(self) -> RealArrayType:  # noqa: N802
        return self.energy_fluence_J_m2 * self.mass_attenuation_m2_kg

    @property
    def dose_rate_Gy_s(self) -> RealArrayType:  # noqa: N802
        return self.energy_fluence_rate_W_m2 * self.mass_attenuation_m2_kg

    @property
    def intensity_W_m2(self) -> RealArrayType:  # noqa: N802
        return self.energy_fluence_rate_W_m2

    def save_npz(self, file_path: Path) -> None:
        numpy.savez_compressed(
            file_path,
            allow_pickle=False,
            photon_number=self.photon_number,
            photon_fluence_1_m2=self.photon_fluence_1_m2,
            photon_fluence_rate_Hz_m2=self.photon_fluence_rate_Hz_m2,
            energy_fluence_J_m2=self.energy_fluence_J_m2,
            energy_fluence_rate_W_m2=self.energy_fluence_rate_W_m2,
            dose_Gy=self.dose_Gy,
            dose_rate_Gy_s=self.dose_rate_Gy_s,
            pixel_height_m=self.pixel_geometry.height_m,
            pixel_width_m=self.pixel_geometry.width_m,
            center_x_m=self.center.coordinate_x_m,
            center_y_m=self.center.coordinate_y_m,
        )


def compute_illumination_map(product: Product) -> IlluminationMap:
    """Build a per-object-pixel photon-count canvas by summing the subpixel-shifted
    probe intensities at every scan position, packaged with the metadata needed to
    derive fluence, dose, and intensity quantities."""
    object_geometry = product.object_.get_geometry()
    probe_geometry = product.probes.get_geometry()
    canvas = numpy.zeros((object_geometry.height_px, object_geometry.width_px))

    for scan_point, probe in product.iter_position_probes():
        object_point = object_geometry.map_coordinates_probe_to_object(scan_point)
        cx = object_point.coordinate_x_px
        cy = object_point.coordinate_y_px

        x_lower = int(cx - probe_geometry.width_px / 2)
        y_lower = int(cy - probe_geometry.height_px / 2)

        dx = cx - (x_lower + probe_geometry.width_px / 2)
        dy = cy - (y_lower + probe_geometry.height_px / 2)

        shifted_modes = fourier_shift_2d(probe.get_array(), dx=dx, dy=dy)
        patch = numpy.sum(numpy.abs(shifted_modes) ** 2, axis=0)
        canvas[
            y_lower : y_lower + probe_geometry.height_px,
            x_lower : x_lower + probe_geometry.width_px,
        ] += patch

    exposure_time_s = product.metadata.exposure_time_s
    photon_flux_Hz = float('nan')  # noqa: N806

    try:
        photon_flux_Hz = product.metadata.probe_photon_count / exposure_time_s  # noqa: N806
    except ZeroDivisionError:
        pass

    return IlluminationMap(
        photon_number=canvas,
        photon_flux_Hz=photon_flux_Hz,
        photon_energy_J=product.metadata.probe_energy_J,
        exposure_time_s=exposure_time_s,
        mass_attenuation_m2_kg=product.metadata.mass_attenuation_m2_kg,
        pixel_geometry=object_geometry.get_pixel_geometry(),
        center=object_geometry.get_center(),
    )

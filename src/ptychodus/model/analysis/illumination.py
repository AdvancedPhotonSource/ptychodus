from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import logging

import numpy

from ptychodus.api.common import RealArrayType
from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.object import ObjectCenter
from ptychodus.api.observer import Observable

from ..product import ProductRepository
from .interpolators import BarycentricArrayStitcher


__all__ = [
    'IlluminationMapper',
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IlluminationMap:
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


class IlluminationMapper(Observable):
    def __init__(self, repository: ProductRepository) -> None:
        super().__init__()
        self._repository = repository

        self._product_index = -1
        self._illumination_map: IlluminationMap | None = None

    def set_product(self, product_index: int) -> None:
        if self._product_index != product_index:
            self._product_index = product_index
            self._illumination_map = None
            self.notify_observers()

    def get_product_name(self) -> str:
        product = self._repository[self._product_index]
        return product.get_name()

    def map(self) -> None:
        product = self._repository[self._product_index].get_product()
        object_geometry = product.object_.get_geometry()

        stitcher = BarycentricArrayStitcher[numpy.floating[Any]](
            numpy.zeros((object_geometry.height_px, object_geometry.width_px))
        )

        for scan_point, probe in zip(product.probe_positions, product.probes):
            object_point = object_geometry.map_coordinates_probe_to_object(scan_point)
            stitcher.add_patch(
                object_point.coordinate_x_px,
                object_point.coordinate_y_px,
                probe.get_intensity(),
            )

        exposure_time_s = product.metadata.exposure_time_s
        photon_flux_Hz = float('nan')  # noqa: N806

        try:
            photon_flux_Hz = product.metadata.probe_photon_count / exposure_time_s  # noqa: N806
        except ZeroDivisionError:
            pass

        self._illumination_map = IlluminationMap(
            photon_number=stitcher.stitch(),
            photon_flux_Hz=photon_flux_Hz,
            photon_energy_J=product.metadata.probe_energy_J,
            exposure_time_s=exposure_time_s,
            mass_attenuation_m2_kg=product.metadata.mass_attenuation_m2_kg,
            pixel_geometry=object_geometry.get_pixel_geometry(),
            center=object_geometry.get_center(),
        )
        self.notify_observers()

    def get_illumination_map(self) -> IlluminationMap:
        if self._illumination_map is None:
            raise ValueError('No analyzed data!')

        return self._illumination_map

    def get_save_file_filters(self) -> Sequence[str]:
        return [self.get_save_file_filter()]

    def get_save_file_filter(self) -> str:
        return 'NumPy Zipped Archive (*.npz)'

    def save_data(self, file_path: Path) -> None:
        if self._illumination_map is None:
            raise ValueError('No analyzed data!')

        contents: dict[str, Any] = {
            'photon_number': self._illumination_map.photon_number,
            'photon_fluence_1_m2': self._illumination_map.photon_fluence_1_m2,
            'photon_fluence_rate_Hz_m2': self._illumination_map.photon_fluence_rate_Hz_m2,
            'energy_fluence_J_m2': self._illumination_map.energy_fluence_J_m2,
            'energy_fluence_rate_W_m2': self._illumination_map.energy_fluence_rate_W_m2,
            'dose_Gy': self._illumination_map.dose_Gy,
            'dose_rate_Gy_s': self._illumination_map.dose_rate_Gy_s,
            'pixel_height_m': self._illumination_map.pixel_geometry.height_m,
            'pixel_width_m': self._illumination_map.pixel_geometry.width_m,
            'center_x_m': self._illumination_map.center.coordinate_x_m,
            'center_y_m': self._illumination_map.center.coordinate_y_m,
        }

        numpy.savez_compressed(file_path, allow_pickle=False, **contents)

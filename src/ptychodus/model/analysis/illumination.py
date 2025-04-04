from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import logging

import numpy

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.object import ObjectCenter
from ptychodus.api.observer import Observable
from ptychodus.api.visualization import RealArrayType

from ..product import ProductRepository
from .barycentric import BarycentricArrayStitcher


__all__ = [
    'IlluminationMapper',
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IlluminationMap:
    photon_number: RealArrayType
    photon_energy_J: float  # noqa: N815
    exposure_time_s: float
    mass_attenuation_m2_kg: float
    pixel_geometry: PixelGeometry | None
    center: ObjectCenter | None

    @property
    def photon_fluence_1_m2(self) -> RealArrayType:
        return (
            numpy.zeros_like(self.photon_number)
            if self.pixel_geometry is None
            else self.photon_number / self.pixel_geometry.area_m2
        )

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
        self._product_data: IlluminationMap | None = None

    def set_product(self, product_index: int) -> None:
        if self._product_index != product_index:
            self._product_index = product_index
            self._product_data = None
            self.notify_observers()

    def get_product_name(self) -> str:
        product = self._repository[self._product_index]
        return product.get_name()

    def map(self) -> None:
        product = self._repository[self._product_index].get_product()
        object_geometry = product.object_.get_geometry()

        stitcher = BarycentricArrayStitcher[numpy.double](
            numpy.zeros((object_geometry.height_px, object_geometry.width_px))
        )

        for scan_point in product.positions:
            object_point = object_geometry.map_scan_point_to_object_point(scan_point)
            stitcher.add_patch(
                object_point.position_x_px,
                object_point.position_y_px,
                product.probe.get_intensity(),  # FIXME OPR & scaling
            )

        self._product_data = IlluminationMap(
            photon_number=stitcher.stitch(),
            photon_energy_J=product.metadata.probe_energy_J,
            exposure_time_s=product.metadata.exposure_time_s,
            mass_attenuation_m2_kg=product.metadata.mass_attenuation_m2_kg,
            pixel_geometry=object_geometry.get_pixel_geometry(),
            center=object_geometry.get_center(),
        )
        self.notify_observers()

    def get_data(self) -> IlluminationMap:
        if self._product_data is None:
            raise ValueError('No analyzed data!')

        return self._product_data

    def get_save_file_filters(self) -> Sequence[str]:
        return [self.get_save_file_filter()]

    def get_save_file_filter(self) -> str:
        return 'NumPy Zipped Archive (*.npz)'

    def save_data(self, file_path: Path) -> None:
        if self._product_data is None:
            raise ValueError('No analyzed data!')

        contents: dict[str, Any] = {
            'photon_number': self._product_data.photon_number,
            'photon_fluence_1_m2': self._product_data.photon_fluence_1_m2,
            'photon_fluence_rate_Hz_m2': self._product_data.photon_fluence_rate_Hz_m2,
            'energy_fluence_J_m2': self._product_data.energy_fluence_J_m2,
            'energy_fluence_rate_W_m2': self._product_data.energy_fluence_rate_W_m2,
            'dose_Gy': self._product_data.dose_Gy,
            'dose_rate_Gy_s': self._product_data.dose_rate_Gy_s,
        }

        pixel_geometry = self._product_data.pixel_geometry

        if pixel_geometry is not None:
            contents['pixel_height_m'] = pixel_geometry.height_m
            contents['pixel_width_m'] = pixel_geometry.width_m

        center = self._product_data.center

        if center is not None:
            contents['center_x_m'] = center.position_x_m
            contents['center_y_m'] = center.position_y_m

        numpy.savez(file_path, **contents)

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
    photon_count: RealArrayType
    photon_flux_Hz: RealArrayType  # noqa: N815
    exposure_J_m2: RealArrayType  # noqa: N815
    irradiance_W_m2: RealArrayType  # noqa: N815
    dose_Gy: RealArrayType  # noqa: N815
    dose_rate_Gy_s: RealArrayType  # noqa: N815
    pixel_geometry: PixelGeometry | None
    center: ObjectCenter | None


class IlluminationMapper(Observable):
    def __init__(self, repository: ProductRepository) -> None:
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

    def map(self) -> IlluminationMap:
        product = self._repository[self._product_index]
        object_item = product.get_object()
        object_ = object_item.get_object()
        object_geometry = object_.get_geometry()

        photon_count = numpy.zeros_like(object_.get_array(), dtype=float)  # FIXME
        photon_flux_Hz = numpy.zeros_like(object_.get_array(), dtype=float)  # FIXME
        exposure_J_m2 = numpy.zeros_like(object_.get_array(), dtype=float)  # FIXME
        irradiance_W_m2 = numpy.zeros_like(object_.get_array(), dtype=float)  # FIXME
        dose_Gy = numpy.zeros_like(object_.get_array(), dtype=float)  # FIXME
        dose_rate_Gy_s = numpy.zeros_like(object_.get_array(), dtype=float)  # FIXME

        return IlluminationMap(
            photon_count=photon_count,
            photon_flux_Hz=photon_flux_Hz,
            exposure_J_m2=exposure_J_m2,
            irradiance_W_m2=irradiance_W_m2,
            dose_Gy=dose_Gy,
            dose_rate_Gy_s=dose_rate_Gy_s,
            pixel_geometry=object_geometry.get_pixel_geometry(),
            center=object_geometry.get_center(),
        )

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
            'photon_count': self._product_data.photon_count,
            'photon_flux_Hz': self._product_data.photon_flux_Hz,
            'exposure_J_m2': self._product_data.exposure_J_m2,
            'irradiance_W_m2': self._product_data.irradiance_W_m2,
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

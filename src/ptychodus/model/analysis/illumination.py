from __future__ import annotations
from collections.abc import Sequence
from pathlib import Path
from typing import Any
import logging

import numpy

from ptychodus.api.illumination import IlluminationMap, compute_illumination_map
from ptychodus.api.observer import Observable

from ..product import ProductRepository


__all__ = [
    'IlluminationMap',
    'IlluminationMapper',
]

logger = logging.getLogger(__name__)


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
        self._illumination_map = compute_illumination_map(product)
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

from __future__ import annotations
from collections.abc import Sequence
from pathlib import Path
from typing import Any
import logging

import numpy

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.observer import Observable
from ptychodus.api.probe import ProbeSequence
from ptychodus.api.propagator import (
    AngularSpectrumPropagator,
    PropagatorParameters,
    ComplexArrayType,
    intensity,
)
from ptychodus.api.typing import RealArrayType

from ..product import ProductRepository
from .settings import ProbePropagationSettings

logger = logging.getLogger(__name__)


class ProbePropagator(Observable):
    def __init__(self, settings: ProbePropagationSettings, repository: ProductRepository) -> None:
        super().__init__()
        self._settings = settings
        self._repository = repository

        self._product_index = -1
        self._propagated_wavefield: ComplexArrayType | None = None
        self._propagated_intensity: RealArrayType | None = None

    def set_product(self, product_index: int) -> None:
        if self._product_index != product_index:
            self._product_index = product_index
            self._propagated_wavefield = None
            self._propagated_intensity = None
            self.notify_observers()

    def get_product_name(self) -> str:
        item = self._repository[self._product_index]
        return item.get_name()

    def propagate(
        self,
        *,
        begin_coordinate_m: float,
        end_coordinate_m: float,
        num_steps: int,
    ) -> None:
        item = self._repository[self._product_index]
        probe = item.get_probe_item().get_probes().get_probe_no_opr()  # TODO OPR
        wavelength_m = item.get_geometry().probe_wavelength_m
        propagated_wavefield = numpy.zeros(
            (num_steps, probe.num_incoherent_modes, probe.height_px, probe.width_px),
            dtype=probe.dtype,
        )
        propagated_intensity = numpy.zeros((num_steps, probe.height_px, probe.width_px))
        distance_m = numpy.linspace(begin_coordinate_m, end_coordinate_m, num_steps)
        pixel_geometry = probe.get_pixel_geometry()

        for idx, z_m in enumerate(distance_m):
            propagator_parameters = PropagatorParameters(
                wavelength_m=wavelength_m,
                width_px=probe.width_px,
                height_px=probe.height_px,
                pixel_width_m=pixel_geometry.width_m,
                pixel_height_m=pixel_geometry.height_m,
                propagation_distance_m=float(z_m),
            )
            propagator = AngularSpectrumPropagator(propagator_parameters)

            for mode in range(probe.num_incoherent_modes):
                wf = propagator.propagate(probe.get_incoherent_mode(mode))
                propagated_wavefield[idx, mode, :, :] = wf
                propagated_intensity[idx, :, :] += intensity(wf)

        self._settings.begin_coordinate_m.set_value(begin_coordinate_m)
        self._settings.end_coordinate_m.set_value(end_coordinate_m)
        self._propagated_wavefield = propagated_wavefield
        self._propagated_intensity = propagated_intensity
        self.notify_observers()

    def get_begin_coordinate_m(self) -> float:
        return self._settings.begin_coordinate_m.get_value()

    def get_end_coordinate_m(self) -> float:
        return self._settings.end_coordinate_m.get_value()

    def _get_probe(self) -> ProbeSequence:
        item = self._repository[self._product_index]
        return item.get_probe_item().get_probes()

    def get_pixel_geometry(self) -> PixelGeometry | None:
        try:
            probe = self._get_probe()
        except IndexError:
            return None
        else:
            return probe.get_pixel_geometry()

    def get_num_steps(self) -> int:
        if self._propagated_intensity is None:
            return self._settings.num_steps.get_value()

        return self._propagated_intensity.shape[0]

    def get_xy_projection(self, step: int) -> RealArrayType:
        if self._propagated_intensity is None:
            raise ValueError('No propagated wavefield!')

        return self._propagated_intensity[step]

    def get_zx_projection(self) -> RealArrayType:
        if self._propagated_intensity is None:
            raise ValueError('No propagated wavefield!')

        sz = self._propagated_intensity.shape[-2]
        cut_plane_l = self._propagated_intensity[:, (sz - 1) // 2, :]
        cut_plane_r = self._propagated_intensity[:, sz // 2, :]
        return numpy.transpose(numpy.add(cut_plane_l, cut_plane_r) / 2)

    def get_zy_projection(self) -> RealArrayType:
        if self._propagated_intensity is None:
            raise ValueError('No propagated wavefield!')

        sz = self._propagated_intensity.shape[-1]
        cut_plane_l = self._propagated_intensity[:, :, (sz - 1) // 2]
        cut_plane_r = self._propagated_intensity[:, :, sz // 2]
        return numpy.transpose(numpy.add(cut_plane_l, cut_plane_r) / 2)

    def get_save_file_filters(self) -> Sequence[str]:
        return [self.get_save_file_filter()]

    def get_save_file_filter(self) -> str:
        return 'NumPy Zipped Archive (*.npz)'

    def save_propagated_probe(self, file_path: Path) -> None:
        if self._propagated_wavefield is None or self._propagated_intensity is None:
            raise ValueError('No propagated wavefield!')

        contents: dict[str, Any] = {
            'begin_coordinate_m': self.get_begin_coordinate_m(),
            'end_coordinate_m': self.get_end_coordinate_m(),
            'wavefield': self._propagated_wavefield,
            'intensity': self._propagated_intensity,
        }

        pixel_geometry = self.get_pixel_geometry()

        if pixel_geometry is not None:
            contents['pixel_height_m'] = pixel_geometry.height_m
            contents['pixel_width_m'] = pixel_geometry.width_m

        numpy.savez_compressed(file_path, allow_pickle=False, **contents)

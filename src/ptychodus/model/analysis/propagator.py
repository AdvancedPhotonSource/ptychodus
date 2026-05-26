from __future__ import annotations
from collections.abc import Sequence
from pathlib import Path
import logging

from ptychodus.api.common import RealArrayType
from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.observer import Observable
from ptychodus.api.probe import ProbeSequence
from ptychodus.api.propagator import PropagatedProbe, propagate_probe

from ..product import ProductRepository
from .settings import ProbePropagationSettings

logger = logging.getLogger(__name__)


class ProbePropagator(Observable):
    def __init__(self, settings: ProbePropagationSettings, repository: ProductRepository) -> None:
        super().__init__()
        self._settings = settings
        self._repository = repository

        self._product_index = -1
        self._result: PropagatedProbe | None = None

    def set_product(self, product_index: int) -> None:
        if self._product_index != product_index:
            self._product_index = product_index
            self._result = None
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
        probes = item.get_probe_item().get_probes()
        wavelength_m = item.get_geometry().probe_wavelength_m

        # OPR caveat: propagate only the first coherent mode and discard any
        # per-position weighting. Matches the long-standing behavior; warn so
        # users with OPR reconstructions are not silently misled.
        if probes.get_opr_weights_or_none() is not None:
            logger.warning(
                'ProbeSequence has OPR weights; propagation uses only the first coherent mode '
                'and discards per-position variation.'
            )

        probe = probes.get_probe_no_opr()

        self._result = propagate_probe(
            probe.get_array(),
            pixel_geometry=probe.get_pixel_geometry(),
            wavelength_m=wavelength_m,
            begin_coordinate_m=begin_coordinate_m,
            end_coordinate_m=end_coordinate_m,
            num_steps=num_steps,
        )
        self._settings.begin_coordinate_m.set_value(begin_coordinate_m)
        self._settings.end_coordinate_m.set_value(end_coordinate_m)
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
        if self._result is None:
            return self._settings.num_steps.get_value()

        return self._result.num_steps

    def get_xy_projection(self, step: int) -> RealArrayType:
        if self._result is None:
            raise ValueError('No propagated wavefield!')

        return self._result.get_xy_projection(step)

    def get_zx_projection(self) -> RealArrayType:
        if self._result is None:
            raise ValueError('No propagated wavefield!')

        return self._result.get_zx_projection()

    def get_zy_projection(self) -> RealArrayType:
        if self._result is None:
            raise ValueError('No propagated wavefield!')

        return self._result.get_zy_projection()

    def get_save_file_filters(self) -> Sequence[str]:
        return [self.get_save_file_filter()]

    def get_save_file_filter(self) -> str:
        return 'NumPy Zipped Archive (*.npz)'

    def save_propagated_probe(self, file_path: Path) -> None:
        if self._result is None:
            raise ValueError('No propagated wavefield!')

        self._result.save_npz(file_path)

from __future__ import annotations
import logging

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.propagator import PropagatedProbe, propagate_probe

from ..product import ProductRepository
from .settings import ProbePropagatorSettings

logger = logging.getLogger(__name__)


class ProbePropagator:
    def __init__(self, settings: ProbePropagatorSettings, repository: ProductRepository) -> None:
        self._settings = settings
        self._repository = repository

    def get_product_name(self, product_index: int) -> str:
        return self._repository[product_index].get_name()

    def get_pixel_geometry(self, product_index: int) -> PixelGeometry | None:
        try:
            item = self._repository[product_index]
        except IndexError:
            return None

        return item.get_probe_item().get_probes().get_pixel_geometry()

    def propagate(self, product_index: int) -> PropagatedProbe:
        item = self._repository[product_index]
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

        return propagate_probe(
            probe.get_array(),
            pixel_geometry=probe.get_pixel_geometry(),
            wavelength_m=wavelength_m,
            begin_coordinate_m=self._settings.begin_coordinate_m.get_value(),
            end_coordinate_m=self._settings.end_coordinate_m.get_value(),
            num_steps=self._settings.num_steps.get_value(),
        )

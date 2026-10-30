from __future__ import annotations
import logging
import math
import time

from ptychodus.api.probe import (
    OPRModeSeries,
    OPRModeStatistics,
    ProbeEnsemble,
    compute_opr_mode_series,
)
from ptychodus.api.probe_positions import ProbePositionSequence
from ptychodus.api.typing import RealArrayType

import numpy

from ..product import ProductRepository


__all__ = [
    'OPRModeAnalyzer',
    'OPRModeSeries',
    'OPRModeStatistics',
]

logger = logging.getLogger(__name__)


class OPRModeAnalyzer:
    def __init__(self, repository: ProductRepository) -> None:
        self._repository = repository

    def get_product_name(self, product_index: int) -> str:
        return self._repository[product_index].get_name()

    def get_probes(self, product_index: int) -> ProbeEnsemble:
        return self._repository[product_index].get_probe_item().get_probes()

    def get_probe_positions(self, product_index: int) -> ProbePositionSequence:
        return self._repository[product_index].get_probe_positions_item().get_probe_positions()

    def get_measured_photon_counts(self, product_index: int) -> RealArrayType | None:
        """Per-position I0, or the detector totals when no I0 was recorded.

        The I0 the probe positions carry is what the incident beam did; without it the
        totals from the bound data's moments are the nearest measure, folding in the
        sample's absorption, and the log says which is returned. None when neither is
        available.

        The counts are keyed by scan index, while OPR weights are addressed by position
        row, so they are re-expressed in row order here. A position whose scan index has
        no count gets NaN rather than being dropped, which keeps the result aligned with
        the weight rows and leaves a gap where a plot would otherwise bridge over
        missing data.
        """
        item = self._repository[product_index]
        positions = item.get_probe_positions_item().get_probe_positions()
        counts_by_index = positions.get_probe_photon_counts_by_index()

        if counts_by_index:
            logger.info('Comparing the OPR weights with the recorded I0.')
        else:
            dataset = item.get_dataset()
            moments = None if dataset is None else dataset.get_diffraction_moments()

            if moments is None:
                return None

            logger.info('No I0 was recorded; comparing the OPR weights with the detector totals.')
            counts_by_index = {
                int(index): float(total)
                for index, total in zip(moments.indexes, moments.total_counts, strict=True)
            }

        return numpy.array(
            [counts_by_index.get(int(position.index), math.nan) for position in positions]
        )

    def analyze(self, product_index: int) -> OPRModeSeries:
        item = self._repository[product_index]
        probes = item.get_probe_item().get_probes()
        num_positions = len(item.get_probe_positions_item().get_probe_positions())

        num_opr_positions = probes.num_opr_positions

        if num_opr_positions is not None and num_opr_positions != num_positions:
            logger.warning(
                'OPR weights describe %d probe position(s) but the product has %d; '
                'per-position quantities can only be paired over the shorter of the two.',
                num_opr_positions,
                num_positions,
            )

        logger.info('Computing OPR mode series...')
        tic = time.perf_counter()
        result = compute_opr_mode_series(probes)
        toc = time.perf_counter()
        logger.info(f'Computed OPR mode series in {toc - tic:.4f} seconds.')

        return result

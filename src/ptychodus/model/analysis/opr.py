from __future__ import annotations
import logging
import math
import time

from ptychodus.api.assemble import compute_probe_photon_counts_by_index
from ptychodus.api.probe import (
    OPRModeSeries,
    OPRModeStatistics,
    ProbeSequence,
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

    def get_probes(self, product_index: int) -> ProbeSequence:
        return self._repository[product_index].get_probe_item().get_probes()

    def get_probe_positions(self, product_index: int) -> ProbePositionSequence:
        return self._repository[product_index].get_probe_positions_item().get_probe_positions()

    def get_measured_photon_counts(self, product_index: int) -> RealArrayType | None:
        """Per-position measured photon counts, or ``None`` when no dataset is bound.

        The counts come back keyed by scan index, while OPR weights are addressed by
        position row, so they are re-expressed in row order here. A position whose scan
        index the dataset does not carry gets NaN rather than being dropped, which keeps
        the result aligned with the weight rows and leaves a gap where a plot would
        otherwise bridge over missing data.
        """
        item = self._repository[product_index]
        dataset = item.get_dataset()

        if dataset is None:
            return None

        positions = item.get_probe_positions_item().get_probe_positions()
        counts_by_index = compute_probe_photon_counts_by_index(
            dataset.get_assembled_data(), positions
        )

        if not counts_by_index:
            return None

        return numpy.array(
            [counts_by_index.get(int(position.index), math.nan) for position in positions]
        )

    def analyze(self, product_index: int) -> OPRModeSeries:
        item = self._repository[product_index]
        probes = item.get_probe_item().get_probes()
        num_positions = len(item.get_probe_positions_item().get_probe_positions())

        if len(probes) != num_positions:
            logger.warning(
                'OPR weights describe %d probe position(s) but the product has %d; '
                'per-position quantities can only be paired over the shorter of the two.',
                len(probes),
                num_positions,
            )

        logger.info('Computing OPR mode series...')
        tic = time.perf_counter()
        result = compute_opr_mode_series(probes)
        toc = time.perf_counter()
        logger.info(f'Computed OPR mode series in {toc - tic:.4f} seconds.')

        return result

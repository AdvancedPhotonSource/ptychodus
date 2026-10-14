from collections.abc import Sequence
import logging
import time

import numpy

from ptychodus.api.affine import AffineFitResult, estimate_affine_transform

from ..product import ProbePositionsRepository
from .settings import AffineTransformEstimatorSettings

logger = logging.getLogger(__name__)


class AffineTransformEstimator:
    def __init__(
        self,
        rng: numpy.random.Generator,
        settings: AffineTransformEstimatorSettings,
        repository: ProbePositionsRepository,
    ) -> None:
        self._rng = rng
        self._settings = settings
        self._repository = repository

    def estimate(
        self,
        measured_product_indexes: Sequence[int],
        corrected_product_indexes: Sequence[int],
    ) -> AffineFitResult:
        """Fit one shared affine transform across every measured/corrected product pair.

        The two index sequences correspond elementwise: entry ``i`` of each names the same scan
        before and after position refinement.
        """
        corrected_set = set(corrected_product_indexes)
        measured_set = set(measured_product_indexes)

        if len(corrected_set) != len(corrected_product_indexes):
            raise ValueError('One or more duplicated corrected product indexes!')

        if len(measured_set) != len(measured_product_indexes):
            raise ValueError('One or more duplicated measured product indexes!')

        if not corrected_set.isdisjoint(measured_set):
            raise ValueError('Product index appears in corrected and measured sets!')

        if len(corrected_product_indexes) != len(measured_product_indexes):
            raise ValueError(
                f'Got {len(measured_product_indexes)} measured product index(es) and '
                f'{len(corrected_product_indexes)} corrected one(s); each measured product needs '
                'the corrected product it pairs with at the same offset.'
            )

        position_pairs = [
            (
                self._repository[measured_index].get_probe_positions(),
                self._repository[corrected_index].get_probe_positions(),
            )
            for measured_index, corrected_index in zip(
                measured_product_indexes, corrected_product_indexes
            )
        ]

        logger.info('Computing affine transform...')
        tic = time.perf_counter()
        result = estimate_affine_transform(
            position_pairs,
            num_iterations=self._settings.num_iterations.get_value(),
            inlier_threshold_m=self._settings.inlier_threshold_m.get_value(),
            min_inliers=self._settings.min_inliers.get_value(),
            rng=self._rng,
        )
        toc = time.perf_counter()
        logger.info(f'Computed affine transform in {toc - tic:.4f} seconds.')
        logger.info(result)

        return result

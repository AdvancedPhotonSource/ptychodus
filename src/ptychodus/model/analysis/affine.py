from collections.abc import Sequence

import numpy

from ptychodus.api.affine import estimate_affine_transform_ransac
from ptychodus.api.geometry import AffineTransform

from ..product import ProbePositionsRepository
from .settings import AffineTransformEstimatorSettings


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
    ) -> AffineTransform:
        corrected_set = set(corrected_product_indexes)
        measured_set = set(measured_product_indexes)

        if len(corrected_set) != len(corrected_product_indexes):
            raise ValueError('One or more duplicated corrected product indexes!')

        if len(measured_set) != len(measured_product_indexes):
            raise ValueError('One or more duplicated measured product indexes!')

        if not corrected_set.isdisjoint(measured_set):
            raise ValueError('Product index appears in corrected and measured sets!')

        measured_positions = [
            self._repository[idx].get_probe_positions() for idx in measured_product_indexes
        ]
        corrected_positions = [
            self._repository[idx].get_probe_positions() for idx in corrected_product_indexes
        ]

        return estimate_affine_transform_ransac(
            measured_positions,
            corrected_positions,
            num_iterations=self._settings.num_iterations.get_value(),
            inlier_threshold=self._settings.inlier_threshold.get_value(),
            min_inliers=self._settings.min_inliers.get_value(),
            rng=self._rng,
        )

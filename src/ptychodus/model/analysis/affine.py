from collections.abc import Sequence
from dataclasses import dataclass

import numpy

from ptychodus.api.geometry import AffineTransform
from ptychodus.api.observer import Observable
from ptychodus.api.typing import RealArrayType

from ..product import ScanRepository
from .settings import AffineTransformEstimatorSettings

__all__ = ['AffineTransformEstimator']


@dataclass(frozen=True)
class PreprocessedCoordinates:
    coordinates: RealArrayType
    centroid_x: float
    centroid_y: float
    rms_distance: float


def estimate_mean_hodges_lehman(values: RealArrayType) -> float:
    mean = numpy.median((values[numpy.newaxis, :] + values[:, numpy.newaxis]) / 2)
    return float(mean)


def estimate_affine_transform(
    uncorrected_coordinates: RealArrayType,
    corrected_coordinates: RealArrayType,
) -> AffineTransform:
    ones_col = numpy.ones((uncorrected_coordinates.shape[0]))
    a = numpy.repeat(numpy.column_stack((uncorrected_coordinates, ones_col)), 2, axis=0)
    b = corrected_coordinates.flatten()
    x, residuals, rank, singular_values = numpy.linalg.lstsq(a, b)
    return AffineTransform(x[0], x[1], x[2], x[3], x[4], x[5])


def evaluate_error(
    uncorrected_coordinates: RealArrayType,
    corrected_coordinates: RealArrayType,
    model: AffineTransform,
) -> RealArrayType:
    y0 = uncorrected_coordinates[-2]
    x0 = uncorrected_coordinates[-1]

    transform = numpy.vectorize(model.__call__)
    yt, xt = transform(y0, x0)

    y1 = corrected_coordinates[-2]
    x1 = corrected_coordinates[-1]

    return numpy.hypot(xt - x1, yt - y1)


class AffineTransformEstimator(Observable):
    def __init__(
        self,
        rng: numpy.random.Generator,
        settings: AffineTransformEstimatorSettings,
        repository: ScanRepository,
    ) -> None:
        self._rng = rng
        self._settings = settings
        self._repository = repository

    def _preprocess_coordinates(self, product_indexes: Sequence[int]) -> PreprocessedCoordinates:
        coordinate_list: list[float] = []

        for product_index in product_indexes:
            positions = self._repository[product_index].get_scan()

            for point in positions:
                coordinate_list.append(point.position_y_m)
                coordinate_list.append(point.position_x_m)

        coordinates = numpy.reshape(coordinate_list, (-1, 2))

        # robust centroid estimation
        centroid_x = estimate_mean_hodges_lehman(coordinates[:, -1])
        centroid_y = estimate_mean_hodges_lehman(coordinates[:, -2])
        coordinates -= numpy.array((centroid_y, centroid_x))

        # rescale for RMS distance = 1
        distance = numpy.hypot(coordinates[:, -1], coordinates[:, -2])
        rms_distance = numpy.sqrt(numpy.mean(numpy.square(distance)))
        coordinates /= rms_distance

        return PreprocessedCoordinates(coordinates, centroid_x, centroid_y, rms_distance)

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

        corrected_coordinates = self._preprocess_coordinates(corrected_product_indexes)
        measured_coordinates = self._preprocess_coordinates(measured_product_indexes)
        indexes = numpy.arange(measured_coordinates.coordinates.shape[0])
        num_shuffles = self._settings.num_shuffles.get_value()
        inlier_threshold = self._settings.inlier_threshold.get_value()
        min_inliers = self._settings.min_inliers.get_value()
        arity = 3  # minimum number of points needed to estimate the model

        best_error = numpy.inf
        best_model = AffineTransform(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)

        # RANSAC estimation of affine transform
        for it in range(num_shuffles):
            self._rng.shuffle(indexes)

            for chunk in range(0, len(indexes), arity):
                samples = indexes[chunk : chunk + arity]

                corrected_subset = numpy.take(corrected_coordinates.coordinates, samples, axis=0)
                uncorrected_subset = numpy.take(measured_coordinates.coordinates, samples, axis=0)
                coarse_model = estimate_affine_transform(uncorrected_subset, corrected_subset)
                error = evaluate_error(uncorrected_subset, corrected_subset, coarse_model)
                inliers = numpy.where(error < inlier_threshold)

                if len(inliers) > min_inliers:
                    corrected_subset = numpy.take(
                        corrected_coordinates.coordinates, inliers, axis=0
                    )
                    uncorrected_subset = numpy.take(
                        measured_coordinates.coordinates, inliers, axis=0
                    )
                    candidate_model = estimate_affine_transform(
                        uncorrected_subset, corrected_subset
                    )
                    candidate_error = evaluate_error(
                        uncorrected_subset, corrected_subset, candidate_model
                    )
                    candidate_error_rms = numpy.sqrt(numpy.mean(numpy.square(candidate_error)))

                    if candidate_error < best_error:
                        best_error = candidate_error_rms
                        best_model = candidate_model

        # TODO broken: unscale best_model

        return best_model

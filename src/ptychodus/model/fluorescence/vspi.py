from __future__ import annotations
from typing import Final
import logging
import time

from scipy.sparse.linalg import lsmr, LinearOperator
import numpy

from ptychodus.api.fluorescence import (
    ElementMap,
    FluorescenceDataset,
    FluorescenceEnhancingAlgorithm,
)
from ptychodus.api.object import ObjectPoint
from ptychodus.api.observer import Observable, Observer
from ptychodus.api.product import Product
from ptychodus.api.typing import RealArrayType

from .settings import FluorescenceSettings

logger = logging.getLogger(__name__)

__all__ = [
    'VSPIFluorescenceEnhancingAlgorithm',
]


class ArrayPatchInterpolator:
    def __init__(self, array: RealArrayType, point: ObjectPoint, shape: tuple[int, ...]) -> None:
        # top left corner of object support
        xmin = point.position_x_px - shape[-1] / 2
        ymin = point.position_y_px - shape[-2] / 2

        # whole components (pixel indexes)
        xmin_wh = int(xmin)
        ymin_wh = int(ymin)

        # fractional (subpixel) components
        xmin_fr = xmin - xmin_wh
        ymin_fr = ymin - ymin_wh

        # bottom right corner of object patch support
        xmax_wh = xmin_wh + shape[-1] + 1
        ymax_wh = ymin_wh + shape[-2] + 1

        # reused quantities
        xmin_fr_c = 1.0 - xmin_fr
        ymin_fr_c = 1.0 - ymin_fr

        # barycentric interpolant weights
        self._weight00 = ymin_fr_c * xmin_fr_c
        self._weight01 = ymin_fr_c * xmin_fr
        self._weight10 = ymin_fr * xmin_fr_c
        self._weight11 = ymin_fr * xmin_fr

        # extract patch support region from full object
        self._support = array[ymin_wh:ymax_wh, xmin_wh:xmax_wh]

    def get_patch(self) -> RealArrayType:
        """interpolate array support to extract patch"""
        patch = self._weight00 * self._support[:-1, :-1]
        patch += self._weight01 * self._support[:-1, 1:]
        patch += self._weight10 * self._support[1:, :-1]
        patch += self._weight11 * self._support[1:, 1:]
        return patch

    def accumulate_patch(self, patch: RealArrayType) -> None:
        """add patch update to array support"""
        self._support[:-1, :-1] += self._weight00 * patch
        self._support[:-1, 1:] += self._weight01 * patch
        self._support[1:, :-1] += self._weight10 * patch
        self._support[1:, 1:] += self._weight11 * patch


class VSPILinearOperator(LinearOperator):
    def __init__(self, product: Product) -> None:
        """
        M: number of XRF positions
        N: number of ptychography object pixels
        P: number of XRF channels

        A[M,N] * X[N,P] = B[M,P]
        """
        object_geometry = product.object_.get_geometry()
        M = len(product.positions)  # noqa: N806
        N = object_geometry.height_px * object_geometry.width_px  # noqa: N806
        super().__init__(float, (M, N))
        self._product = product

    def _matvec(self, x: RealArrayType) -> RealArrayType:  # noqa: N803
        object_geometry = self._product.object_.get_geometry()
        object_array = x.reshape((object_geometry.height_px, object_geometry.width_px))
        AX = numpy.zeros(len(self._product.positions))  # noqa: N806

        for index, (scan_point, probe) in enumerate(
            zip(self._product.positions, self._product.probes)
        ):
            object_point = object_geometry.map_scan_point_to_object_point(scan_point)
            probe_intensity = probe.get_intensity()
            psf = probe_intensity / probe_intensity.sum()
            interpolator = ArrayPatchInterpolator(object_array, object_point, psf.shape)
            AX[index] = numpy.sum(psf * interpolator.get_patch())

        return AX

    def _rmatvec(self, x: RealArrayType) -> RealArrayType:  # noqa: N803
        object_geometry = self._product.object_.get_geometry()
        object_array = numpy.zeros((object_geometry.height_px, object_geometry.width_px))

        for index, (scan_point, probe) in enumerate(
            zip(self._product.positions, self._product.probes)
        ):
            object_point = object_geometry.map_scan_point_to_object_point(scan_point)
            probe_intensity = probe.get_intensity()
            psf = probe_intensity / probe_intensity.sum()
            interpolator = ArrayPatchInterpolator(object_array, object_point, psf.shape)
            interpolator.accumulate_patch(x[index] * psf)

        HX = object_array.flatten()  # noqa: N806

        return HX


class VSPIFluorescenceEnhancingAlgorithm(FluorescenceEnhancingAlgorithm, Observable, Observer):
    SIMPLE_NAME: Final[str] = 'VSPI'
    DISPLAY_NAME: Final[str] = 'Virtual Single Pixel Imaging'

    def __init__(self, settings: FluorescenceSettings) -> None:
        super().__init__()
        self._settings = settings

        settings.vspi_damping_factor.add_observer(self)
        settings.vspi_max_iterations.add_observer(self)

    def enhance(self, dataset: FluorescenceDataset, product: Product) -> FluorescenceDataset:
        object_geometry = product.object_.get_geometry()
        e_cps_shape = object_geometry.height_px, object_geometry.width_px
        element_maps: list[ElementMap] = list()
        A = VSPILinearOperator(product)  # noqa: N806

        for emap in dataset.element_maps:
            logger.info(f'Enhancing "{emap.name}"...')
            tic = time.perf_counter()
            m_cps = emap.counts_per_second
            result = lsmr(
                A,
                m_cps.flatten(),
                damp=self._settings.vspi_damping_factor.get_value(),
                maxiter=self._settings.vspi_max_iterations.get_value(),
                show=True,
            )
            logger.debug(result)
            e_cps = result[0].reshape(e_cps_shape)
            emap_enhanced = ElementMap(emap.name, e_cps)
            toc = time.perf_counter()
            logger.info(f'Enhanced "{emap.name}" in {toc - tic:.4f} seconds.')

            element_maps.append(emap_enhanced)

        return FluorescenceDataset(
            element_maps=element_maps,
            counts_per_second_path=dataset.counts_per_second_path,
            channel_names_path=dataset.channel_names_path,
        )

    def get_damping_factor(self) -> float:
        return self._settings.vspi_damping_factor.get_value()

    def set_damping_factor(self, factor: float) -> None:
        self._settings.vspi_damping_factor.set_value(factor)

    def get_max_iterations(self) -> int:
        return self._settings.vspi_max_iterations.get_value()

    def set_max_iterations(self, number: int) -> None:
        self._settings.vspi_max_iterations.set_value(number)

    def _update(self, observable: Observable) -> None:
        if observable is self._settings.vspi_damping_factor:
            self.notify_observers()
        elif observable is self._settings.vspi_max_iterations:
            self.notify_observers()

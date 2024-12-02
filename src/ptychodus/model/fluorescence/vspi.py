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
        xmin = point.positionXInPixels - shape[-1] / 2
        ymin = point.positionYInPixels - shape[-2] / 2

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
        object_geometry = product.object_.getGeometry()
        M = len(product.scan)
        N = object_geometry.heightInPixels * object_geometry.widthInPixels
        super().__init__(float, (M, N))
        self._product = product

    def _get_psf(self) -> RealArrayType:
        intensity = self._product.probe.getIntensity()
        return intensity / intensity.sum()

    def _matvec(self, X: RealArrayType) -> RealArrayType:
        object_geometry = self._product.object_.getGeometry()
        object_array = X.reshape((object_geometry.heightInPixels, object_geometry.widthInPixels))
        psf = self._get_psf()
        AX = numpy.zeros(len(self._product.scan))

        for index, scan_point in enumerate(self._product.scan):
            object_point = object_geometry.mapScanPointToObjectPoint(scan_point)
            interpolator = ArrayPatchInterpolator(object_array, object_point, psf.shape)
            AX[index] = numpy.sum(psf * interpolator.get_patch())

        return AX

    def _rmatvec(self, X: RealArrayType) -> RealArrayType:
        object_geometry = self._product.object_.getGeometry()
        object_array = numpy.zeros((object_geometry.heightInPixels, object_geometry.widthInPixels))
        psf = self._get_psf()

        for index, scan_point in enumerate(self._product.scan):
            object_point = object_geometry.mapScanPointToObjectPoint(scan_point)
            interpolator = ArrayPatchInterpolator(object_array, object_point, psf.shape)
            interpolator.accumulate_patch(X[index] * psf)

        HX = object_array.flatten()

        return HX


class VSPIFluorescenceEnhancingAlgorithm(FluorescenceEnhancingAlgorithm, Observable, Observer):
    SIMPLE_NAME: Final[str] = 'VSPI'
    DISPLAY_NAME: Final[str] = 'Virtual Single Pixel Imaging'

    def __init__(self, settings: FluorescenceSettings) -> None:
        super().__init__()
        self._settings = settings

        settings.vspiDampingFactor.addObserver(self)
        settings.vspiMaxIterations.addObserver(self)

    def enhance(self, dataset: FluorescenceDataset, product: Product) -> FluorescenceDataset:
        # FIXME OPR
        object_geometry = product.object_.getGeometry()
        e_cps_shape = object_geometry.heightInPixels, object_geometry.widthInPixels
        element_maps: list[ElementMap] = list()
        A = VSPILinearOperator(product)

        for emap in dataset.element_maps:
            logger.info(f'Enhancing "{emap.name}"...')
            tic = time.perf_counter()
            m_cps = emap.counts_per_second
            result = lsmr(
                A,
                m_cps.flatten(),
                damp=self._settings.vspiDampingFactor.getValue(),
                maxiter=self._settings.vspiMaxIterations.getValue(),
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

    def getDampingFactor(self) -> float:
        return self._settings.vspiDampingFactor.getValue()

    def setDampingFactor(self, factor: float) -> None:
        self._settings.vspiDampingFactor.setValue(factor)

    def getMaxIterations(self) -> int:
        return self._settings.vspiMaxIterations.getValue()

    def setMaxIterations(self, number: int) -> None:
        self._settings.vspiMaxIterations.setValue(number)

    def update(self, observable: Observable) -> None:
        if observable is self._settings.vspiDampingFactor:
            self.notifyObservers()
        elif observable is self._settings.vspiMaxIterations:
            self.notifyObservers()

from __future__ import annotations
from collections.abc import Sequence
from pathlib import Path
from typing import Final
import logging
import time

from scipy.sparse.linalg import lsmr, LinearOperator
import numpy

from ptychodus.api.fluorescence import (
    DeconvolutionStrategy,
    ElementMap,
    FluorescenceDataset,
    FluorescenceFileReader,
    FluorescenceFileWriter,
    UpscalingStrategy,
)
from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.object import ObjectPoint
from ptychodus.api.observer import Observable, Observer
from ptychodus.api.plugins import PluginChooser
from ptychodus.api.product import Product
from ptychodus.api.typing import RealArrayType

from ..product import ProductRepository
from .settings import FluorescenceSettings

logger = logging.getLogger(__name__)


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


class FluorescenceEnhancer(Observable, Observer):
    VSPI: Final[str] = 'Virtual Single Pixel Imaging'
    TWO_STEP: Final[str] = 'Upscale and Deconvolve'

    def __init__(
        self,
        settings: FluorescenceSettings,
        productRepository: ProductRepository,
        upscalingStrategyChooser: PluginChooser[UpscalingStrategy],
        deconvolutionStrategyChooser: PluginChooser[DeconvolutionStrategy],
        fileReaderChooser: PluginChooser[FluorescenceFileReader],
        fileWriterChooser: PluginChooser[FluorescenceFileWriter],
        reinitObservable: Observable,
    ) -> None:
        super().__init__()
        self._settings = settings
        self._productRepository = productRepository
        self._upscalingStrategyChooser = upscalingStrategyChooser
        self._deconvolutionStrategyChooser = deconvolutionStrategyChooser
        self._fileReaderChooser = fileReaderChooser
        self._fileWriterChooser = fileWriterChooser
        self._reinitObservable = reinitObservable

        self._productIndex = -1
        self._measured: FluorescenceDataset | None = None
        self._enhanced: FluorescenceDataset | None = None

        upscalingStrategyChooser.addObserver(self)
        upscalingStrategyChooser.setCurrentPluginByName(settings.upscalingStrategy.getValue())
        deconvolutionStrategyChooser.addObserver(self)
        deconvolutionStrategyChooser.setCurrentPluginByName(
            settings.deconvolutionStrategy.getValue()
        )
        fileReaderChooser.setCurrentPluginByName(settings.fileType.getValue())
        fileWriterChooser.setCurrentPluginByName(settings.fileType.getValue())
        reinitObservable.addObserver(self)

    def setProduct(self, productIndex: int) -> None:
        if self._productIndex != productIndex:
            self._productIndex = productIndex
            self._enhanced = None
            self.notifyObservers()

    def getProductName(self) -> str:
        return self._productRepository[self._productIndex].getName()

    def getOpenFileFilterList(self) -> Sequence[str]:
        return self._fileReaderChooser.getDisplayNameList()

    def getOpenFileFilter(self) -> str:
        return self._fileReaderChooser.currentPlugin.displayName

    def openMeasuredDataset(self, filePath: Path, fileFilter: str) -> None:
        if filePath.is_file():
            self._fileReaderChooser.setCurrentPluginByName(fileFilter)
            fileType = self._fileReaderChooser.currentPlugin.simpleName
            logger.debug(f'Reading "{filePath}" as "{fileType}"')
            fileReader = self._fileReaderChooser.currentPlugin.strategy

            try:
                measured = fileReader.read(filePath)
            except Exception as exc:
                raise RuntimeError(f'Failed to read "{filePath}"') from exc
            else:
                self._measured = measured
                self._enhanced = None

                self._settings.filePath.setValue(filePath)
                self._settings.fileType.setValue(fileType)

                self.notifyObservers()
        else:
            logger.warning(f'Refusing to load dataset from invalid file path "{filePath}"')

    def getNumberOfChannels(self) -> int:
        return 0 if self._measured is None else len(self._measured.element_maps)

    def getMeasuredElementMap(self, channelIndex: int) -> ElementMap:
        if self._measured is None:
            raise ValueError('Fluorescence dataset not loaded!')

        return self._measured.element_maps[channelIndex]

    def getEnhancementStrategyList(self) -> Sequence[str]:
        return [self.VSPI, self.TWO_STEP]

    def getEnhancementStrategy(self) -> str:
        return self.VSPI if self._settings.useVSPI.getValue() else self.TWO_STEP

    def setEnhancementStrategy(self, name: str) -> None:
        self._settings.useVSPI.setValue(name.casefold() == self.VSPI.casefold())

    def getUpscalingStrategyList(self) -> Sequence[str]:
        return self._upscalingStrategyChooser.getDisplayNameList()

    def getUpscalingStrategy(self) -> str:
        return self._upscalingStrategyChooser.currentPlugin.displayName

    def setUpscalingStrategy(self, name: str) -> None:
        self._upscalingStrategyChooser.setCurrentPluginByName(name)

    def getDeconvolutionStrategyList(self) -> Sequence[str]:
        return self._deconvolutionStrategyChooser.getDisplayNameList()

    def getDeconvolutionStrategy(self) -> str:
        return self._deconvolutionStrategyChooser.currentPlugin.displayName

    def setDeconvolutionStrategy(self, name: str) -> None:
        self._deconvolutionStrategyChooser.setCurrentPluginByName(name)

    def enhanceFluorescence(self) -> None:
        if self._measured is None:
            raise ValueError('Fluorescence dataset not loaded!')

        product = self._productRepository[self._productIndex].getProduct()
        object_geometry = product.object_.getGeometry()
        e_cps_shape = object_geometry.heightInPixels, object_geometry.widthInPixels
        element_maps: list[ElementMap] = list()

        if self._settings.useVSPI.getValue():
            A = VSPILinearOperator(product)

            for emap in self._measured.element_maps:
                logger.info(f'Enhancing "{emap.name}"...')
                tic = time.perf_counter()
                m_cps = emap.counts_per_second
                result = lsmr(A, m_cps.flatten(), maxiter=100, show=True)  # TODO expose parameters
                logger.debug(result)
                e_cps = result[0].reshape(e_cps_shape)
                emap_enhanced = ElementMap(emap.name, e_cps)
                toc = time.perf_counter()
                logger.info(f'Enhanced "{emap.name}" in {toc - tic:.4f} seconds.')

                element_maps.append(emap_enhanced)
        else:
            upscaler = self._upscalingStrategyChooser.currentPlugin.strategy
            deconvolver = self._deconvolutionStrategyChooser.currentPlugin.strategy

            for emap in self._measured.element_maps:
                logger.info(f'Enhancing "{emap.name}"...')
                tic = time.perf_counter()
                emap_upscaled = upscaler(emap, product)
                emap_enhanced = deconvolver(emap_upscaled, product)
                toc = time.perf_counter()
                logger.info(f'Enhanced "{emap.name}" in {toc - tic:.4f} seconds.')

                element_maps.append(emap_enhanced)

        self._enhanced = FluorescenceDataset(
            element_maps=element_maps,
            counts_per_second_path=self._measured.counts_per_second_path,
            channel_names_path=self._measured.channel_names_path,
        )
        self.notifyObservers()

    def getPixelGeometry(self) -> PixelGeometry:
        return self._productRepository[self._productIndex].getGeometry().getPixelGeometry()

    def getEnhancedElementMap(self, channelIndex: int) -> ElementMap:
        if self._enhanced is None:
            raise ValueError('Fluorescence dataset not enhanced!')

        return self._enhanced.element_maps[channelIndex]

    def getSaveFileFilterList(self) -> Sequence[str]:
        return self._fileWriterChooser.getDisplayNameList()

    def getSaveFileFilter(self) -> str:
        return self._fileWriterChooser.currentPlugin.displayName

    def saveEnhancedDataset(self, filePath: Path, fileFilter: str) -> None:
        if self._enhanced is None:
            raise ValueError('Fluorescence dataset not enhanced!')

        self._fileWriterChooser.setCurrentPluginByName(fileFilter)
        fileType = self._fileWriterChooser.currentPlugin.simpleName
        logger.debug(f'Writing "{filePath}" as "{fileType}"')
        writer = self._fileWriterChooser.currentPlugin.strategy
        writer.write(filePath, self._enhanced)

    def _openFluorescenceFileFromSettings(self) -> None:
        self.openMeasuredDataset(
            self._settings.filePath.getValue(), self._settings.fileType.getValue()
        )

    def update(self, observable: Observable) -> None:
        if observable is self._reinitObservable:
            self._openFluorescenceFileFromSettings()
        elif observable is self._upscalingStrategyChooser:
            strategy = self._upscalingStrategyChooser.currentPlugin.simpleName
            self._settings.upscalingStrategy.setValue(strategy)
            self.notifyObservers()
        elif observable is self._deconvolutionStrategyChooser:
            strategy = self._deconvolutionStrategyChooser.currentPlugin.simpleName
            self._settings.deconvolutionStrategy.setValue(strategy)
            self.notifyObservers()

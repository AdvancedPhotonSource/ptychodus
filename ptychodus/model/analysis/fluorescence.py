from __future__ import annotations
from collections.abc import Sequence
from pathlib import Path
from typing import Final
import logging

from numpy.typing import ArrayLike
from scipy.sparse.linalg import gmres, LinearOperator
import numpy

from ptychodus.api.fluorescence import (DeconvolutionStrategy, ElementMap, FluorescenceDataset,
                                        FluorescenceFileReader, FluorescenceFileWriter,
                                        UpscalingStrategy)
from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.observer import Observable, Observer
from ptychodus.api.plugins import PluginChooser
from ptychodus.api.product import Product
from ptychodus.api.typing import RealArrayType

from ..reconstructor import DiffractionPatternPositionMatcher
from .settings import FluorescenceSettings

logger = logging.getLogger(__name__)


def get_axis_weights_and_indexes(xmin_o: float, dx_o: float, xmin_p: float,
                                 dx_p: float) -> tuple[Sequence[float], Sequence[int]]:
    pos: list[float] = []
    idx: list[int] = []

    # TODO n_p is probe.widthInPixels or probe.heightInPixels
    # TODO frac, whole = numpy.modf(i)
    # TODO foo.ravel()

    x_o = xmin_o
    x_p = xmin_p

    i_o = 0  # FIXME calculate

    while True:  # FIXME exit?
        idx.append(i_o)

        if x_p < x_o:
            pos.append(x_p)

            x_p += dx_p
        elif x_p > x_o:
            pos.append(x_o)

            i_o += 1
            x_o += dx_o
        else:  # x_p == x_o
            pos.append(x_o)

            i_o += 1
            x_o += dx_o
            x_p += dx_p

    return numpy.diff(pos), idx


class VSPILinearOperator(LinearOperator):

    def __init__(self, product: Product, xrf_nchannels: int) -> None:
        '''
        M: number of XRF positions
        N: number of ptychography object pixels
        P: number of XRF channels

        A[M,N] * X[N,P] = B[M,P]
        '''
        super().__init__(float, (len(product.scan), xrf_nchannels))
        self._product = product

    def matmat(self, x: ArrayLike) -> RealArrayType:
        AX = numpy.zeros(self.shape, dtype=self.dtype)

        probeGeometry = self._product.probe.getGeometry()
        dx_p = probeGeometry.pixelWidthInMeters
        dy_p = probeGeometry.pixelHeightInMeters

        objectGeometry = self._product.object_.getGeometry()
        xmin_o = objectGeometry.minimumXInMeters
        ymin_o = objectGeometry.minimumYInMeters
        dx_o = objectGeometry.pixelWidthInMeters
        dy_o = objectGeometry.pixelHeightInMeters

        for index, point in enumerate(self._product.scan):
            xmin_p = point.positionXInMeters - probeGeometry.widthInMeters / 2
            ymin_p = point.positionYInMeters - probeGeometry.heightInMeters / 2

            wx, ix = get_axis_weights_and_indexes(xmin_o, dx_o, xmin_p, dx_p)
            wy, iy = get_axis_weights_and_indexes(ymin_o, dy_o, ymin_p, dy_p)

            i_nz = numpy.meshgrid(iy, ix)  # FIXME

            w_nz = numpy.outer(wy, wx) / (dx_p * dy_p)
            i_nz = numpy.ravel_multi_index(multi_index, dims)  # FIXME
            X_nz = X.take(i_nz, axis=0)

            AX[index, :] = numpy.dot(w_nz.ravel(), X_nz)

        return AX


class FluorescenceEnhancer(Observable, Observer):
    VSPI: Final[str] = 'Virtual Single Pixel Imaging'
    TWO_STEP: Final[str] = 'Upscale and Deconvolve'

    def __init__(
            self,
            settings: FluorescenceSettings,
            dataMatcher: DiffractionPatternPositionMatcher,  # FIXME match XRF too
            upscalingStrategyChooser: PluginChooser[UpscalingStrategy],
            deconvolutionStrategyChooser: PluginChooser[DeconvolutionStrategy],
            fileReaderChooser: PluginChooser[FluorescenceFileReader],
            fileWriterChooser: PluginChooser[FluorescenceFileWriter],
            reinitObservable: Observable) -> None:
        super().__init__()
        self._settings = settings
        self._dataMatcher = dataMatcher
        self._upscalingStrategyChooser = upscalingStrategyChooser
        self._deconvolutionStrategyChooser = deconvolutionStrategyChooser
        self._fileReaderChooser = fileReaderChooser
        self._fileWriterChooser = fileWriterChooser
        self._reinitObservable = reinitObservable

        self._productIndex = -1
        self._measured: FluorescenceDataset | None = None
        self._enhanced: FluorescenceDataset | None = None

        upscalingStrategyChooser.addObserver(self)
        upscalingStrategyChooser.setCurrentPluginByName(settings.upscalingStrategy.value)
        deconvolutionStrategyChooser.addObserver(self)
        deconvolutionStrategyChooser.setCurrentPluginByName(settings.deconvolutionStrategy.value)
        fileReaderChooser.setCurrentPluginByName(settings.fileType.value)
        fileWriterChooser.setCurrentPluginByName(settings.fileType.value)
        reinitObservable.addObserver(self)

    def setProduct(self, productIndex: int) -> None:
        if self._productIndex != productIndex:
            self._productIndex = productIndex
            self._enhanced = None
            self.notifyObservers()

    def getProductName(self) -> str:
        return self._dataMatcher.getProductName(self._productIndex)

    def getOpenFileFilterList(self) -> Sequence[str]:
        return self._fileReaderChooser.getDisplayNameList()

    def getOpenFileFilter(self) -> str:
        return self._fileReaderChooser.currentPlugin.displayName

    def openMeasuredDataset(self, filePath: Path, fileFilter: str) -> None:
        if filePath.is_file():
            self._fileReaderChooser.setCurrentPluginByName(fileFilter)
            fileType = self._fileReaderChooser.currentPlugin.simpleName
            logger.debug(f'Reading \"{filePath}\" as \"{fileType}\"')
            fileReader = self._fileReaderChooser.currentPlugin.strategy

            try:
                measured = fileReader.read(filePath)
            except Exception as exc:
                raise RuntimeError(f'Failed to read \"{filePath}\"') from exc
            else:
                self._measured = measured
                self._enhanced = None

                self._settings.filePath.value = filePath
                self._settings.fileType.value = fileType

                self.notifyObservers()
        else:
            logger.warning(f'Refusing to load dataset from invalid file path \"{filePath}\"')

    def getNumberOfChannels(self) -> int:
        return 0 if self._measured is None else len(self._measured.element_maps)

    def getMeasuredElementMap(self, channelIndex: int) -> ElementMap:
        if self._measured is None:
            raise ValueError('Fluorescence dataset not loaded!')

        return self._measured.element_maps[channelIndex]

    def getEnhancementStrategyList(self) -> Sequence[str]:
        return [self.VSPI, self.TWO_STEP]

    def getEnhancementStrategy(self) -> str:
        return self.VSPI if self._settings.useVSPI.value else self.TWO_STEP

    def setEnhancementStrategy(self, name: str) -> None:
        self._settings.useVSPI.value = (name.casefold() == self.VSPI.casefold())

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

        reconstructInput = self._dataMatcher.matchDiffractionPatternsWithPositions(
            self._productIndex)
        element_maps: list[ElementMap] = list()

        if self._settings.useVSPI.value:
            emaps = self._measured.element_maps
            A = VSPILinearOperator(reconstructInput.product, len(emaps))
            B = numpy.stack([b.counts_per_second.flatten() for b in emaps]).T
            X, exitCode = gmres(A, B, atol=1e-5)

            if exitCode != 0:
                raise RuntimeError() # FIXME print(exitCode) # 0 indicates successful convergence

            # FIXME element_maps.append(emap_enhanced)
        else:
            upscaler = self._upscalingStrategyChooser.currentPlugin.strategy
            deconvolver = self._deconvolutionStrategyChooser.currentPlugin.strategy

            for emap in self._measured.element_maps:
                logger.info(f'Enhancing \"{emap.name}\"')
                emap_upscaled = upscaler(emap, reconstructInput.product)
                emap_enhanced = deconvolver(emap_upscaled, reconstructInput.product)
                element_maps.append(emap_enhanced)

        self._enhanced = FluorescenceDataset(
            element_maps=element_maps,
            counts_per_second_path=self._measured.counts_per_second_path,
            channel_names_path=self._measured.channel_names_path,
        )
        self.notifyObservers()

    def getPixelGeometry(self) -> PixelGeometry:
        return self._dataMatcher.getObjectPlanePixelGeometry(self._productIndex)

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
        logger.debug(f'Writing \"{filePath}\" as \"{fileType}\"')
        writer = self._fileWriterChooser.currentPlugin.strategy
        writer.write(filePath, self._enhanced)

    def _openFluorescenceFileFromSettings(self) -> None:
        self.openMeasuredDataset(self._settings.filePath.value, self._settings.fileType.value)

    def update(self, observable: Observable) -> None:
        if observable is self._reinitObservable:
            self._openFluorescenceFileFromSettings()
        elif observable is self._upscalingStrategyChooser:
            strategy = self._upscalingStrategyChooser.currentPlugin.simpleName
            self._settings.upscalingStrategy.value = strategy
            self.notifyObservers()
        elif observable is self._deconvolutionStrategyChooser:
            strategy = self._deconvolutionStrategyChooser.currentPlugin.simpleName
            self._settings.deconvolutionStrategy.value = strategy
            self.notifyObservers()

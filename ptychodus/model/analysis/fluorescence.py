from __future__ import annotations
from collections.abc import Sequence
from pathlib import Path
from typing import Final
import logging

from scipy.sparse.linalg import gmres, LinearOperator
import math
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
from ptychodus.api.observer import Observable, Observer
from ptychodus.api.plugins import PluginChooser
from ptychodus.api.product import Product
from ptychodus.api.typing import RealArrayType

from ..reconstructor import DiffractionPatternPositionMatcher
from .settings import FluorescenceSettings

logger = logging.getLogger(__name__)


def get_axis_weights_and_indexes(xmin_o: float, dx_o: float, xmin_p: float, dx_p: float,
                                 N_p: int) -> tuple[Sequence[float], Sequence[int]]:
    weight: list[float] = []
    index: list[int] = []

    x_l = xmin_p
    n_o = math.ceil((x_l - xmin_o) / dx_o)

    for n_p in range(N_p):
        x_p = xmin_p + (n_p + 1) * dx_p

        while True:
            x_o = xmin_o + n_o + dx_o

            if x_o >= x_p:
                break

            weight.append((x_o - x_l) / dx_p)
            index.append(n_o)

            n_o += 1
            x_l = x_o

        weight.append((x_p - x_l) / dx_p)
        index.append(n_o)
        x_l = x_p

        if x_o == x_p:
            n_o += 1

    return weight, index


class VSPILinearOperator(LinearOperator):

    def __init__(self, product: Product, xrf_nchannels: int) -> None:
        """
        M: number of XRF positions
        N: number of ptychography object pixels
        P: number of XRF channels

        A[M,N] * X[N,P] = B[M,P]
        """
        super().__init__(float, (len(product.scan), xrf_nchannels))
        self._product = product

    def matmat(self, X: RealArrayType) -> RealArrayType:
        AX = numpy.zeros(self.shape, dtype=self.dtype)

        probeGeometry = self._product.probe.getGeometry()
        dx_p_m = probeGeometry.pixelWidthInMeters
        dy_p_m = probeGeometry.pixelHeightInMeters

        objectGeometry = self._product.object_.getGeometry()
        objectShape = objectGeometry.heightInPixels, objectGeometry.widthInPixels
        xmin_o_m = objectGeometry.minimumXInMeters
        ymin_o_m = objectGeometry.minimumYInMeters
        dx_o_m = objectGeometry.pixelWidthInMeters
        dy_o_m = objectGeometry.pixelHeightInMeters

        for index, point in enumerate(self._product.scan):
            xmin_p_m = point.positionXInMeters - probeGeometry.widthInMeters / 2
            ymin_p_m = point.positionYInMeters - probeGeometry.heightInMeters / 2

            wx, ix = get_axis_weights_and_indexes(xmin_o_m, dx_o_m, xmin_p_m, dx_p_m,
                                                  probeGeometry.widthInPixels)
            wy, iy = get_axis_weights_and_indexes(ymin_o_m, dy_o_m, ymin_p_m, dy_p_m,
                                                  probeGeometry.heightInPixels)

            IY, IX = numpy.meshgrid(iy, ix)
            i_nz = numpy.ravel_multi_index(list(zip(IY.flat, IX.flat)), objectShape)
            X_nz = X.take(i_nz, axis=0)

            AX[index, :] = numpy.matmul(numpy.outer(wy, wx).ravel(), X_nz)

        return AX


class FluorescenceEnhancer(Observable, Observer):
    VSPI: Final[str] = "Virtual Single Pixel Imaging"
    TWO_STEP: Final[str] = "Upscale and Deconvolve"

    def __init__(
        self,
        settings: FluorescenceSettings,
        dataMatcher: DiffractionPatternPositionMatcher,  # FIXME match XRF too
        upscalingStrategyChooser: PluginChooser[UpscalingStrategy],
        deconvolutionStrategyChooser: PluginChooser[DeconvolutionStrategy],
        fileReaderChooser: PluginChooser[FluorescenceFileReader],
        fileWriterChooser: PluginChooser[FluorescenceFileWriter],
        reinitObservable: Observable,
    ) -> None:
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
        upscalingStrategyChooser.setCurrentPluginByName(settings.upscalingStrategy.getValue())
        deconvolutionStrategyChooser.addObserver(self)
        deconvolutionStrategyChooser.setCurrentPluginByName(
            settings.deconvolutionStrategy.getValue())
        fileReaderChooser.setCurrentPluginByName(settings.fileType.getValue())
        fileWriterChooser.setCurrentPluginByName(settings.fileType.getValue())
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
            raise ValueError("Fluorescence dataset not loaded!")

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
            raise ValueError("Fluorescence dataset not loaded!")

        reconstructInput = self._dataMatcher.matchDiffractionPatternsWithPositions(
            self._productIndex)
        element_maps: list[ElementMap] = list()

        if self._settings.useVSPI.getValue():
            measured_emaps = self._measured.element_maps
            A = VSPILinearOperator(reconstructInput.product, len(measured_emaps))
            B = numpy.stack([b.counts_per_second.flatten() for b in measured_emaps]).T
            X, info = gmres(A, B, atol=1e-5)  # TODO expose atol

            if info != 0:
                logger.warning(f"Convergence to tolerance not achieved! {info=}")

            for m_emap, e_cps in zip(measured_emaps, X.T):
                e_emap = ElementMap(m_emap.name, e_cps.reshape(m_emap.counts_per_second.shape))
                element_maps.append(e_emap)

        else:
            upscaler = self._upscalingStrategyChooser.currentPlugin.strategy
            deconvolver = self._deconvolutionStrategyChooser.currentPlugin.strategy

            for emap in self._measured.element_maps:
                logger.info(f'Enhancing "{emap.name}"')
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
            raise ValueError("Fluorescence dataset not enhanced!")

        return self._enhanced.element_maps[channelIndex]

    def getSaveFileFilterList(self) -> Sequence[str]:
        return self._fileWriterChooser.getDisplayNameList()

    def getSaveFileFilter(self) -> str:
        return self._fileWriterChooser.currentPlugin.displayName

    def saveEnhancedDataset(self, filePath: Path, fileFilter: str) -> None:
        if self._enhanced is None:
            raise ValueError("Fluorescence dataset not enhanced!")

        self._fileWriterChooser.setCurrentPluginByName(fileFilter)
        fileType = self._fileWriterChooser.currentPlugin.simpleName
        logger.debug(f'Writing "{filePath}" as "{fileType}"')
        writer = self._fileWriterChooser.currentPlugin.strategy
        writer.write(filePath, self._enhanced)

    def _openFluorescenceFileFromSettings(self) -> None:
        self.openMeasuredDataset(self._settings.filePath.getValue(),
                                 self._settings.fileType.getValue())

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

from __future__ import annotations
from collections.abc import Sequence
from pathlib import Path
import logging

from ptychodus.api.fluorescence import (DeconvolutionStrategy, ElementMap, FluorescenceDataset,
                                        FluorescenceFileReader, FluorescenceFileWriter,
                                        UpscalingStrategy)
from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.observer import Observable, Observer
from ptychodus.api.plugins import PluginChooser

from ..reconstructor import DiffractionPatternPositionMatcher
from .settings import FluorescenceSettings

logger = logging.getLogger(__name__)


class FluorescenceEnhancer(Observable, Observer):

    def __init__(self, settings: FluorescenceSettings,
                 dataMatcher: DiffractionPatternPositionMatcher,
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

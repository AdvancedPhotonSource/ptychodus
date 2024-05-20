from __future__ import annotations
from collections.abc import Sequence
from pathlib import Path
import logging

from ptychodus.api.fluorescence import (DeconvolutionStrategy, ElementMap, FluorescenceDataset,
                                        FluorescenceFileReader, FluorescenceFileWriter,
                                        UpscalingStrategy)
from ptychodus.api.observer import Observable, Observer
from ptychodus.api.plugins import PluginChooser

from ..product import ProductRepository
from .settings import FluorescenceSettings

logger = logging.getLogger(__name__)


class FluorescenceEnhancer(Observable, Observer):

    def __init__(self, settings: FluorescenceSettings, repository: ProductRepository,
                 upscalingStrategyChooser: PluginChooser[UpscalingStrategy],
                 deconvolutionStrategyChooser: PluginChooser[DeconvolutionStrategy],
                 fileReaderChooser: PluginChooser[FluorescenceFileReader],
                 fileWriterChooser: PluginChooser[FluorescenceFileWriter],
                 reinitObservable: Observable) -> None:
        self._settings = settings
        self._repository = repository
        # FIXME vvv get/set & sync to/from settings vvv
        self._upscalingStrategyChooser = upscalingStrategyChooser
        self._deconvolutionStrategyChooser = deconvolutionStrategyChooser
        # FIXME ^^^^^
        self._fileReaderChooser = fileReaderChooser
        self._fileWriterChooser = fileWriterChooser
        self._reinitObservable = reinitObservable
        self._measured: FluorescenceDataset | None = None
        self._enhanced: FluorescenceDataset | None = None

        reinitObservable.addObserver(self)
        self._syncFromSettings()

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
        else:
            logger.warning(f'Refusing to load dataset from invalid file path \"{filePath}\"')

    def getNumberOfChannels(self) -> int:
        return 0 if self._measured is None else len(self._measured.element_maps)

    def getMeasuredElementMap(self, channelIndex: int) -> ElementMap:
        if self._measured is None:
            raise ValueError('Fluorescence dataset not loaded!')

        return self._measured.element_maps[channelIndex]

    def enhanceFluorescence(self, itemIndex: int) -> None:
        if self._measured is None:
            logger.debug('Fluorescence dataset not loaded!')
            return

        element_maps: list[ElementMap] = list()
        product = self._repository[itemIndex].getProduct()
        upscaler = self._upscalingStrategyChooser.currentPlugin.strategy
        deconvolver = self._deconvolutionStrategyChooser.currentPlugin.strategy

        for emap in self._measured.element_maps:
            logger.debug(f'Processing \"{emap.name}\"')
            emap_upscaled = upscaler(emap, product)
            emap_enhanced = deconvolver(emap_upscaled, product)
            element_maps.append(emap_enhanced)

        self._enhanced = FluorescenceDataset(
            element_maps=element_maps,
            counts_per_second_path=self._measured.counts_per_second_path,
            channel_names_path=self._measured.channel_names_path,
        )

    def getEnhancedElementMap(self, channelIndex: int) -> ElementMap:
        if self._enhanced is None:
            raise ValueError('Fluorescence dataset not enhanced!')

        return self._enhanced.element_maps[channelIndex]

    def getSaveFileFilterList(self) -> Sequence[str]:
        return self._fileWriterChooser.getDisplayNameList()

    def getSaveFileFilter(self) -> str:
        return self._fileWriterChooser.currentPlugin.displayName

    def saveEnhancedDataset(self, filePath: Path, fileFilter: str) -> None:
        self._fileWriterChooser.setCurrentPluginByName(fileFilter)
        fileType = self._fileWriterChooser.currentPlugin.simpleName
        logger.debug(f'Writing \"{filePath}\" as \"{fileType}\"')
        writer = self._fileWriterChooser.currentPlugin.strategy

        if self._enhanced is None:
            raise ValueError('Fluorescence dataset not enhanced!')
        else:
            writer.write(filePath, self._enhanced)

    def _syncFromSettings(self) -> None:
        self._upscalingStrategyChooser.setCurrentPluginByName(
            self._settings.upscalingStrategy.value)
        self._deconvolutionStrategyChooser.setCurrentPluginByName(
            self._settings.deconvolutionStrategy.value)
        self._fileReaderChooser.setCurrentPluginByName(self._settings.fileType.value)
        self._fileWriterChooser.setCurrentPluginByName(self._settings.fileType.value)

    def update(self, observable: Observable) -> None:
        if observable is self._reinitObservable:
            self._syncFromSettings()

from __future__ import annotations
from collections.abc import Iterator, Sequence
from pathlib import Path
import logging

from ptychodus.api.plugins import PluginChooser
from ptychodus.api.xrf import (DeconvolutionStrategy, ElementMap, FluorescenceDataset,
                               FluorescenceFileReader, FluorescenceFileWriter, UpscalingStrategy)

from ..product import ProductRepository

logger = logging.getLogger(__name__)


class FluorescenceEnhancer:

    def __init__(self, repository: ProductRepository,
                 upscalingStrategyChooser: PluginChooser[UpscalingStrategy],
                 deconvolutionStrategyChooser: PluginChooser[DeconvolutionStrategy],
                 fileReaderChooser: PluginChooser[FluorescenceFileReader],
                 fileWriterChooser: PluginChooser[FluorescenceFileWriter]) -> None:
        # FIXME defaults from settings
        self._repository = repository
        self._upscalingStrategyChooser = upscalingStrategyChooser
        self._deconvolutionStrategyChooser = deconvolutionStrategyChooser
        self._fileReaderChooser = fileReaderChooser
        self._fileWriterChooser = fileWriterChooser
        self._measured: FluorescenceDataset | None = None
        self._enhanced: FluorescenceDataset | None = None

    def getOpenFileFilterList(self) -> Sequence[str]:
        return self._fileReaderChooser.getDisplayNameList()

    def getOpenFileFilter(self) -> str:
        return self._fileReaderChooser.currentPlugin.displayName

    def openMeasuredFluorescenceDataset(self, filePath: Path, fileFilter: str) -> None:
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
        else:
            logger.warning(f'Refusing to create product with invalid file path \"{filePath}\"')

    def channels(self) -> Iterator[str]:
        if self._measured is not None:
            for channel in self._measured.element_maps.keys():
                yield channel

    def enhanceXRF(self, itemIndex: int) -> None:
        if self._measured is None:
            raise ValueError('Fluorescence dataset not loaded!')

        # FIXME BEGIN
        product = self._repository[itemIndex]
        # FIXME self._upscalingStrategyChooser.setCurrentPluginByName(self._settings.upscaling_strategy.value)
        # FIXME self._deconvolutionStrategyChooser.setCurrentPluginByName(self._settings.deconvolution_strategy.value)
        element_maps: dict[str, ElementMap] = dict()
        upscaler = self._upscalingStrategyChooser.currentPlugin.strategy
        deconvolver = self._deconvolutionStrategyChooser.currentPlugin.strategy

        for channelName, element_map in xrf.element_maps.items():
            logger.debug(f'Processing \"{channelName}\"')
            emap_upscaled = upscaler(element_map, ptycho)
            emap_enhanced = deconvolver(emap_upscaled, ptycho)
            element_maps[channelName] = emap_enhanced
            plot_emap(Path(f'{channelName}.png'), element_map, emap_upscaled, emap_enhanced)

        xrf_enhanced = FluorescenceDataset(
            element_maps=element_maps,
            counts_per_second_path=xrf.counts_per_second_path,
            channel_names_path=xrf.channelNames_path,
        )
        # FIXME self._enhanced = enhanced
        # FIXME END

    def getMeasuredElementMap(self, channel: str) -> ElementMap:
        if self._measured is None:
            raise ValueError('Fluorescence dataset not loaded!')

        return self._measured.element_maps[channel]

    def getEnhancedElementMap(self, channel: str, itemIndex: int) -> ElementMap:
        if self._enhanced is None:
            raise ValueError('Fluorescence dataset not enhanced!')

        return self._enhanced.element_maps[channel]

    def getSaveFileFilterList(self) -> Sequence[str]:
        return self._fileWriterChooser.getDisplayNameList()

    def getSaveFileFilter(self) -> str:
        return self._fileWriterChooser.currentPlugin.displayName

    def saveEnhancedFluorescenceDataset(self, filePath: Path, fileFilter: str) -> None:
        self._fileWriterChooser.setCurrentPluginByName(fileFilter)
        fileType = self._fileWriterChooser.currentPlugin.simpleName
        logger.debug(f'Writing \"{filePath}\" as \"{fileType}\"')
        writer = self._fileWriterChooser.currentPlugin.strategy

        if self._enhanced is None:
            raise ValueError('Fluorescence dataset not enhanced!')
        else:
            writer.write(filePath, self._enhanced)

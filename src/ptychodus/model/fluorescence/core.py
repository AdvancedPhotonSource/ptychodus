from __future__ import annotations
from collections.abc import Iterator
from pathlib import Path
import logging


from ptychodus.api.fluorescence import (
    DeconvolutionStrategy,
    ElementMap,
    FluorescenceDataset,
    FluorescenceEnhancingAlgorithm,
    FluorescenceFileReader,
    FluorescenceFileWriter,
    UpscalingStrategy,
)
from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.observer import Observable, Observer
from ptychodus.api.plugins import PluginChooser
from ptychodus.api.settings import SettingsRegistry

from ..product import ProductRepository, ProductRepositoryItem
from ..visualization import VisualizationEngine
from .settings import FluorescenceSettings
from .two_step import TwoStepFluorescenceEnhancingAlgorithm
from .vspi import VSPIFluorescenceEnhancingAlgorithm

logger = logging.getLogger(__name__)


class FluorescenceEnhancer(Observable, Observer):
    def __init__(
        self,
        settings: FluorescenceSettings,
        productRepository: ProductRepository,
        twoStepEnhancingAlgorithm: TwoStepFluorescenceEnhancingAlgorithm,
        vspiEnhancingAlgorithm: VSPIFluorescenceEnhancingAlgorithm,
        fileReaderChooser: PluginChooser[FluorescenceFileReader],
        fileWriterChooser: PluginChooser[FluorescenceFileWriter],
        reinitObservable: Observable,
    ) -> None:
        super().__init__()
        self._settings = settings
        self._productRepository = productRepository
        self.twoStepEnhancingAlgorithm = twoStepEnhancingAlgorithm
        self.vspiEnhancingAlgorithm = vspiEnhancingAlgorithm
        self._fileReaderChooser = fileReaderChooser
        self._fileWriterChooser = fileWriterChooser
        self._reinitObservable = reinitObservable

        self._algorithmChooser = PluginChooser[FluorescenceEnhancingAlgorithm]()
        self._algorithmChooser.register_plugin(
            twoStepEnhancingAlgorithm,
            simple_name=TwoStepFluorescenceEnhancingAlgorithm.SIMPLE_NAME,
            display_name=TwoStepFluorescenceEnhancingAlgorithm.DISPLAY_NAME,
        )
        self._algorithmChooser.register_plugin(
            vspiEnhancingAlgorithm,
            simple_name=VSPIFluorescenceEnhancingAlgorithm.SIMPLE_NAME,
            display_name=VSPIFluorescenceEnhancingAlgorithm.DISPLAY_NAME,
        )
        self._algorithmChooser.synchronize_with_parameter(settings.algorithm)
        self._algorithmChooser.add_observer(self)

        self._productIndex = -1
        self._measured: FluorescenceDataset | None = None
        self._enhanced: FluorescenceDataset | None = None

        fileReaderChooser.synchronize_with_parameter(settings.fileType)
        fileWriterChooser.set_current_plugin(settings.fileType.get_value())
        reinitObservable.add_observer(self)

    @property
    def _product(self) -> ProductRepositoryItem:
        return self._productRepository[self._productIndex]

    def setProduct(self, productIndex: int) -> None:
        if self._productIndex != productIndex:
            self._productIndex = productIndex
            self._enhanced = None
            self.notify_observers()

    def getProductName(self) -> str:
        return self._product.getName()

    def getPixelGeometry(self) -> PixelGeometry:
        return self._product.getGeometry().getObjectPlanePixelGeometry()

    def getOpenFileFilterList(self) -> Iterator[str]:
        for plugin in self._fileReaderChooser:
            yield plugin.display_name

    def getOpenFileFilter(self) -> str:
        return self._fileReaderChooser.get_current_plugin().display_name

    def openMeasuredDataset(self, filePath: Path, fileFilter: str) -> None:
        if filePath.is_file():
            self._fileReaderChooser.set_current_plugin(fileFilter)
            fileType = self._fileReaderChooser.get_current_plugin().simple_name
            logger.debug(f'Reading "{filePath}" as "{fileType}"')
            fileReader = self._fileReaderChooser.get_current_plugin().strategy

            try:
                measured = fileReader.read(filePath)
            except Exception as exc:
                raise RuntimeError(f'Failed to read "{filePath}"') from exc
            else:
                self._measured = measured
                self._enhanced = None

                self._settings.filePath.set_value(filePath)

                self.notify_observers()
        else:
            logger.warning(f'Refusing to load dataset from invalid file path "{filePath}"')

    def getNumberOfChannels(self) -> int:
        return 0 if self._measured is None else len(self._measured.element_maps)

    def getMeasuredElementMap(self, channelIndex: int) -> ElementMap:
        if self._measured is None:
            raise ValueError('Fluorescence dataset not loaded!')

        return self._measured.element_maps[channelIndex]

    def getAlgorithmList(self) -> Iterator[str]:
        for plugin in self._algorithmChooser:
            yield plugin.display_name

    def getAlgorithm(self) -> str:
        return self._algorithmChooser.get_current_plugin().display_name

    def setAlgorithm(self, name: str) -> None:
        self._algorithmChooser.set_current_plugin(name)

    def enhanceFluorescence(self) -> None:
        if self._measured is None:
            raise ValueError('Fluorescence dataset not loaded!')
        else:
            algorithm = self._algorithmChooser.get_current_plugin().strategy
            product = self._product.getProduct()
            self._enhanced = algorithm.enhance(self._measured, product)
            self.notify_observers()

    def getEnhancedElementMap(self, channelIndex: int) -> ElementMap:
        if self._enhanced is None:
            return self.getMeasuredElementMap(channelIndex)

        return self._enhanced.element_maps[channelIndex]

    def getSaveFileFilterList(self) -> Iterator[str]:
        for plugin in self._fileWriterChooser:
            yield plugin.display_name

    def getSaveFileFilter(self) -> str:
        return self._fileWriterChooser.get_current_plugin().display_name

    def saveEnhancedDataset(self, filePath: Path, fileFilter: str) -> None:
        if self._enhanced is None:
            raise ValueError('Fluorescence dataset not enhanced!')

        self._fileWriterChooser.set_current_plugin(fileFilter)
        fileType = self._fileWriterChooser.get_current_plugin().simple_name
        logger.debug(f'Writing "{filePath}" as "{fileType}"')
        writer = self._fileWriterChooser.get_current_plugin().strategy
        writer.write(filePath, self._enhanced)

    def _openFluorescenceFileFromSettings(self) -> None:
        self.openMeasuredDataset(
            self._settings.filePath.get_value(), self._settings.fileType.get_value()
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._algorithmChooser:
            self.notify_observers()
        elif observable is self._reinitObservable:
            self._openFluorescenceFileFromSettings()


class FluorescenceCore:
    def __init__(
        self,
        settingsRegistry: SettingsRegistry,
        productRepository: ProductRepository,
        upscalingStrategyChooser: PluginChooser[UpscalingStrategy],
        deconvolutionStrategyChooser: PluginChooser[DeconvolutionStrategy],
        fileReaderChooser: PluginChooser[FluorescenceFileReader],
        fileWriterChooser: PluginChooser[FluorescenceFileWriter],
    ) -> None:
        self._settings = FluorescenceSettings(settingsRegistry)
        self._twoStepEnhancingAlgorithm = TwoStepFluorescenceEnhancingAlgorithm(
            self._settings, upscalingStrategyChooser, deconvolutionStrategyChooser
        )
        self._vspiEnhancingAlgorithm = VSPIFluorescenceEnhancingAlgorithm(self._settings)

        self.enhancer = FluorescenceEnhancer(
            self._settings,
            productRepository,
            self._twoStepEnhancingAlgorithm,
            self._vspiEnhancingAlgorithm,
            fileReaderChooser,
            fileWriterChooser,
            settingsRegistry,
        )
        self.visualizationEngine = VisualizationEngine(isComplex=False)

    def enhanceFluorescence(
        self, productIndex: int, inputFilePath: Path, outputFilePath: Path
    ) -> int:
        fileType = 'XRF-Maps'

        try:
            self.enhancer.setProduct(productIndex)
            self.enhancer.openMeasuredDataset(inputFilePath, fileType)
            self.enhancer.enhanceFluorescence()
            self.enhancer.saveEnhancedDataset(outputFilePath, fileType)
        except Exception as exc:
            logger.exception(exc)
            return -1

        return 0

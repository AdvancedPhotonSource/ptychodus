from __future__ import annotations
from collections.abc import Sequence
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
        self._algorithmChooser.registerPlugin(
            twoStepEnhancingAlgorithm,
            simpleName=TwoStepFluorescenceEnhancingAlgorithm.SIMPLE_NAME,
            displayName=TwoStepFluorescenceEnhancingAlgorithm.DISPLAY_NAME,
        )
        self._algorithmChooser.registerPlugin(
            vspiEnhancingAlgorithm,
            simpleName=VSPIFluorescenceEnhancingAlgorithm.SIMPLE_NAME,
            displayName=VSPIFluorescenceEnhancingAlgorithm.DISPLAY_NAME,
        )
        self._syncAlgorithmFromSettings()
        self._algorithmChooser.addObserver(self)

        self._productIndex = -1
        self._measured: FluorescenceDataset | None = None
        self._enhanced: FluorescenceDataset | None = None

        fileReaderChooser.setCurrentPluginByName(settings.fileType.getValue())
        fileWriterChooser.setCurrentPluginByName(settings.fileType.getValue())
        reinitObservable.addObserver(self)

    @property
    def _product(self) -> ProductRepositoryItem:
        return self._productRepository[self._productIndex]

    def setProduct(self, productIndex: int) -> None:
        if self._productIndex != productIndex:
            self._productIndex = productIndex
            self._enhanced = None
            self.notifyObservers()

    def getProductName(self) -> str:
        return self._product.getName()

    def getPixelGeometry(self) -> PixelGeometry:
        return self._product.getGeometry().getPixelGeometry()

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

    def getAlgorithmList(self) -> Sequence[str]:
        return self._algorithmChooser.getDisplayNameList()

    def getAlgorithm(self) -> str:
        return self._algorithmChooser.currentPlugin.displayName

    def setAlgorithm(self, name: str) -> None:
        self._algorithmChooser.setCurrentPluginByName(name)
        self._settings.algorithm.setValue(self._algorithmChooser.currentPlugin.simpleName)

    def _syncAlgorithmFromSettings(self) -> None:
        self.setAlgorithm(self._settings.algorithm.getValue())

    def enhanceFluorescence(self) -> None:
        if self._measured is None:
            raise ValueError('Fluorescence dataset not loaded!')
        else:
            algorithm = self._algorithmChooser.currentPlugin.strategy
            product = self._product.getProduct()
            self._enhanced = algorithm.enhance(self._measured, product)
            self.notifyObservers()

    def getEnhancedElementMap(self, channelIndex: int) -> ElementMap:
        if self._enhanced is None:
            return self.getMeasuredElementMap(channelIndex)

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
        if observable is self._algorithmChooser:
            self.notifyObservers()
        elif observable is self._reinitObservable:
            self._syncAlgorithmFromSettings()
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
            self._settings, upscalingStrategyChooser, deconvolutionStrategyChooser, settingsRegistry
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

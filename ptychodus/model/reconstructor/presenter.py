from collections.abc import Sequence
from pathlib import Path
import logging
import time

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.plugins import PluginChooser
from ptychodus.api.reconstructor import Reconstructor, TrainableReconstructor, TrainOutput

from ..product import ProductRepository
from .matcher import DiffractionPatternPositionMatcher, ScanIndexFilter
from .settings import ReconstructorSettings

logger = logging.getLogger(__name__)


class ReconstructorPresenter(Observable, Observer):

    def __init__(self, settings: ReconstructorSettings,
                 dataMatcher: DiffractionPatternPositionMatcher,
                 productRepository: ProductRepository,
                 reconstructorChooser: PluginChooser[Reconstructor],
                 reinitObservable: Observable) -> None:
        super().__init__()
        self._settings = settings
        self._dataMatcher = dataMatcher
        self._productRepository = productRepository
        self._reconstructorChooser = reconstructorChooser
        self._reinitObservable = reinitObservable

        reinitObservable.addObserver(self)
        self._syncFromSettings()

    def getReconstructorList(self) -> Sequence[str]:
        return self._reconstructorChooser.getDisplayNameList()

    def getReconstructor(self) -> str:
        return self._reconstructorChooser.currentPlugin.displayName

    def setReconstructor(self, name: str) -> None:
        self._reconstructorChooser.setCurrentPluginByName(name)
        self._settings.algorithm.value = self._reconstructorChooser.currentPlugin.simpleName
        self.notifyObservers()

    def _syncFromSettings(self) -> None:
        self.setReconstructor(self._settings.algorithm.value)

    def reconstruct(self,
                    inputProductIndex: int,
                    outputProductName: str,
                    indexFilter: ScanIndexFilter = ScanIndexFilter.ALL) -> int:
        reconstructor = self._reconstructorChooser.currentPlugin.strategy
        parameters = self._dataMatcher.matchDiffractionPatternsWithPositions(
            inputProductIndex, indexFilter)

        tic = time.perf_counter()
        result = reconstructor.reconstruct(parameters)
        toc = time.perf_counter()
        logger.info(f'Reconstruction time {toc - tic:.4f} seconds. (code={result.result})')

        outputProductIndex = self._productRepository.insertProduct(result.product)
        return outputProductIndex

    def reconstructSplit(self, inputProductIndex: int, outputProductName: str) -> tuple[int, int]:
        outputProductIndexOdd = self.reconstruct(
            inputProductIndex,
            f'{outputProductName} - Odd',
            ScanIndexFilter.ODD,
        )
        outputProductIndexEven = self.reconstruct(
            inputProductIndex,
            f'{outputProductName} - Even',
            ScanIndexFilter.EVEN,
        )

        return outputProductIndexOdd, outputProductIndexEven

    @property
    def isTrainable(self) -> bool:
        reconstructor = self._reconstructorChooser.currentPlugin.strategy
        return isinstance(reconstructor, TrainableReconstructor)

    def ingestTrainingData(self, inputProductIndex: int) -> None:
        reconstructor = self._reconstructorChooser.currentPlugin.strategy

        if isinstance(reconstructor, TrainableReconstructor):
            logger.info('Preparing input data...')
            tic = time.perf_counter()
            parameters = self._dataMatcher.matchDiffractionPatternsWithPositions(
                inputProductIndex, ScanIndexFilter.ALL)
            toc = time.perf_counter()
            logger.info(f'Data preparation time {toc - tic:.4f} seconds.')

            logger.info('Ingesting...')
            tic = time.perf_counter()
            reconstructor.ingestTrainingData(parameters)
            toc = time.perf_counter()
            logger.info(f'Ingest time {toc - tic:.4f} seconds.')
        else:
            logger.warning('Reconstructor is not trainable!')

    def getOpenTrainingDataFileFilterList(self) -> Sequence[str]:
        reconstructor = self._reconstructorChooser.currentPlugin.strategy

        if isinstance(reconstructor, TrainableReconstructor):
            return reconstructor.getOpenTrainingDataFileFilterList()
        else:
            logger.warning('Reconstructor is not trainable!')

        return list()

    def getOpenTrainingDataFileFilter(self) -> str:
        reconstructor = self._reconstructorChooser.currentPlugin.strategy

        if isinstance(reconstructor, TrainableReconstructor):
            return reconstructor.getOpenTrainingDataFileFilter()
        else:
            logger.warning('Reconstructor is not trainable!')

        return str()

    def openTrainingData(self, filePath: Path) -> None:
        reconstructor = self._reconstructorChooser.currentPlugin.strategy

        if isinstance(reconstructor, TrainableReconstructor):
            logger.info('Opening training data...')
            tic = time.perf_counter()
            reconstructor.openTrainingData(filePath)
            toc = time.perf_counter()
            logger.info(f'Open time {toc - tic:.4f} seconds.')
        else:
            logger.warning('Reconstructor is not trainable!')

    def getSaveTrainingDataFileFilterList(self) -> Sequence[str]:
        reconstructor = self._reconstructorChooser.currentPlugin.strategy

        if isinstance(reconstructor, TrainableReconstructor):
            return reconstructor.getSaveTrainingDataFileFilterList()
        else:
            logger.warning('Reconstructor is not trainable!')

        return list()

    def getSaveTrainingDataFileFilter(self) -> str:
        reconstructor = self._reconstructorChooser.currentPlugin.strategy

        if isinstance(reconstructor, TrainableReconstructor):
            return reconstructor.getSaveTrainingDataFileFilter()
        else:
            logger.warning('Reconstructor is not trainable!')

        return str()

    def saveTrainingData(self, filePath: Path) -> None:
        reconstructor = self._reconstructorChooser.currentPlugin.strategy

        if isinstance(reconstructor, TrainableReconstructor):
            logger.info('Saving training data...')
            tic = time.perf_counter()
            reconstructor.saveTrainingData(filePath)
            toc = time.perf_counter()
            logger.info(f'Save time {toc - tic:.4f} seconds.')
        else:
            logger.warning('Reconstructor is not trainable!')

    def train(self) -> TrainOutput:
        reconstructor = self._reconstructorChooser.currentPlugin.strategy
        result = TrainOutput([], [], -1)

        if isinstance(reconstructor, TrainableReconstructor):
            logger.info('Training...')
            tic = time.perf_counter()
            result = reconstructor.train()
            toc = time.perf_counter()
            logger.info(f'Training time {toc - tic:.4f} seconds. (code={result.result})')
        else:
            logger.warning('Reconstructor is not trainable!')

        return result

    def clearTrainingData(self) -> None:
        reconstructor = self._reconstructorChooser.currentPlugin.strategy

        if isinstance(reconstructor, TrainableReconstructor):
            logger.info('Resetting...')
            tic = time.perf_counter()
            reconstructor.clearTrainingData()
            toc = time.perf_counter()
            logger.info(f'Reset time {toc - tic:.4f} seconds.')
        else:
            logger.warning('Reconstructor is not trainable!')

    def getOpenModelFileFilterList(self) -> Sequence[str]:
        reconstructor = self._reconstructorChooser.currentPlugin.strategy

        if isinstance(reconstructor, TrainableReconstructor):
            return reconstructor.getOpenModelFileFilterList()
        else:
            logger.warning('Reconstructor is not trainable!')

        return list()

    def getOpenModelFileFilter(self) -> str:
        reconstructor = self._reconstructorChooser.currentPlugin.strategy

        if isinstance(reconstructor, TrainableReconstructor):
            return reconstructor.getOpenModelFileFilter()
        else:
            logger.warning('Reconstructor is not trainable!')

        return str()

    def openModel(self, filePath: Path) -> None:
        reconstructor = self._reconstructorChooser.currentPlugin.strategy

        if isinstance(reconstructor, TrainableReconstructor):
            logger.info('Opening model...')
            tic = time.perf_counter()
            reconstructor.openModel(filePath)
            toc = time.perf_counter()
            logger.info(f'Open time {toc - tic:.4f} seconds.')
        else:
            logger.warning('Reconstructor is not trainable!')

    def getSaveModelFileFilterList(self) -> Sequence[str]:
        reconstructor = self._reconstructorChooser.currentPlugin.strategy

        if isinstance(reconstructor, TrainableReconstructor):
            return reconstructor.getSaveModelFileFilterList()
        else:
            logger.warning('Reconstructor is not trainable!')

        return list()

    def getSaveModelFileFilter(self) -> str:
        reconstructor = self._reconstructorChooser.currentPlugin.strategy

        if isinstance(reconstructor, TrainableReconstructor):
            return reconstructor.getSaveModelFileFilter()
        else:
            logger.warning('Reconstructor is not trainable!')

        return str()

    def saveModel(self, filePath: Path) -> None:
        reconstructor = self._reconstructorChooser.currentPlugin.strategy

        if isinstance(reconstructor, TrainableReconstructor):
            logger.info('Saving model...')
            tic = time.perf_counter()
            reconstructor.saveModel(filePath)
            toc = time.perf_counter()
            logger.info(f'Save time {toc - tic:.4f} seconds.')
        else:
            logger.warning('Reconstructor is not trainable!')

    def update(self, observable: Observable) -> None:
        if observable is self._reinitObservable:
            self._syncFromSettings()

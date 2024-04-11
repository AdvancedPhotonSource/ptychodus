from __future__ import annotations
from collections.abc import Sequence
from pathlib import Path
import logging
import time

import numpy

from ...api.observer import Observable, Observer
from ...api.plugins import PluginChooser
from ...api.product import Product
from ...api.reconstructor import (ReconstructInput, Reconstructor, TrainableReconstructor,
                                  TrainOutput)
from ...api.scan import Scan, ScanPoint
from ..patterns import ActiveDiffractionDataset
from ..product import ProductRepository
from .indexFilter import ScanIndexFilter
from .settings import ReconstructorSettings

logger = logging.getLogger(__name__)


class ReconstructorPresenter(Observable, Observer):

    def __init__(self, settings: ReconstructorSettings,
                 diffractionDataset: ActiveDiffractionDataset,
                 productRepository: ProductRepository,
                 reconstructorChooser: PluginChooser[Reconstructor],
                 reinitObservable: Observable) -> None:
        super().__init__()
        self._settings = settings
        self._diffractionDataset = diffractionDataset
        self._productRepository = productRepository
        self._reconstructorChooser = reconstructorChooser
        self._reinitObservable = reinitObservable

    @classmethod
    def createInstance(cls, settings: ReconstructorSettings,
                       diffractionDataset: ActiveDiffractionDataset,
                       productRepository: ProductRepository,
                       reconstructorChooser: PluginChooser[Reconstructor],
                       reinitObservable: Observable) -> ReconstructorPresenter:
        activeReconstructor = cls(settings, diffractionDataset, productRepository,
                                  reconstructorChooser, reinitObservable)
        reinitObservable.addObserver(activeReconstructor)
        activeReconstructor._syncFromSettings()
        return activeReconstructor

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

    def _prepareInputData(self, inputProductIndex: int,
                          indexFilter: ScanIndexFilter) -> ReconstructInput:
        inputProductItem = self._productRepository[inputProductIndex]
        inputProduct = inputProductItem.getProduct()
        dataIndexes = self._diffractionDataset.getAssembledIndexes()
        scanIndexes = [point.index for point in inputProduct.scan if indexFilter(point.index)]
        commonIndexes = sorted(set(dataIndexes).intersection(scanIndexes))

        patterns = numpy.take(
            self._diffractionDataset.getAssembledData(),
            commonIndexes,
            axis=0,
        )

        pointList: list[ScanPoint] = list()
        pointIter = iter(inputProduct.scan)

        for index in commonIndexes:
            while True:
                point = next(pointIter)

                if point.index == index:
                    pointList.append(point)
                    break

        probe = inputProduct.probe  # TODO remap if needed

        product = Product(
            metadata=inputProduct.metadata,
            scan=Scan(pointList),
            probe=probe,
            object_=inputProduct.object_,
            costs=inputProduct.costs,
        )

        return ReconstructInput(patterns, product)

    def reconstruct(self,
                    inputProductIndex: int,
                    outputProductName: str,
                    indexFilter: ScanIndexFilter = ScanIndexFilter.ALL) -> int:
        reconstructor = self._reconstructorChooser.currentPlugin.strategy
        parameters = self._prepareInputData(inputProductIndex, indexFilter)

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
            parameters = self._prepareInputData(inputProductIndex, ScanIndexFilter.ALL)
            toc = time.perf_counter()
            logger.info(f'Data preparation time {toc - tic:.4f} seconds.')

            logger.info('Ingesting...')
            tic = time.perf_counter()
            reconstructor.ingestTrainingData(parameters)
            toc = time.perf_counter()
            logger.info(f'Ingest time {toc - tic:.4f} seconds.')
        else:
            logger.warning('Reconstructor is not trainable!')

    def getSaveFileFilterList(self) -> Sequence[str]:
        reconstructor = self._reconstructorChooser.currentPlugin.strategy

        if isinstance(reconstructor, TrainableReconstructor):
            return reconstructor.getSaveFileFilterList()
        else:
            logger.warning('Reconstructor is not trainable!')

        return list()

    def getSaveFileFilter(self) -> str:
        reconstructor = self._reconstructorChooser.currentPlugin.strategy

        if isinstance(reconstructor, TrainableReconstructor):
            return reconstructor.getSaveFileFilter()
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

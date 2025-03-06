from pathlib import Path
import logging
import time

from ptychodus.api.plugins import PluginChooser
from ptychodus.api.reconstructor import (
    Reconstructor,
    TrainableReconstructor,
    TrainOutput,
)

from ..product import ProductRepository
from .matcher import DiffractionPatternPositionMatcher, ScanIndexFilter
from .queue import ReconstructionQueue

logger = logging.getLogger(__name__)


class ReconstructorAPI:
    def __init__(
        self,
        reconstructionQueue: ReconstructionQueue,
        dataMatcher: DiffractionPatternPositionMatcher,
        productRepository: ProductRepository,
        reconstructorChooser: PluginChooser[Reconstructor],
    ) -> None:
        self._reconstructionQueue = reconstructionQueue
        self._dataMatcher = dataMatcher
        self._productRepository = productRepository
        self._reconstructorChooser = reconstructorChooser

    @property
    def isReconstructing(self) -> bool:
        return self._reconstructionQueue.isReconstructing

    def processResults(self, *, block: bool) -> None:
        self._reconstructionQueue.processResults(block=block)

    def reconstruct(
        self,
        inputProductIndex: int,
        *,
        outputProductSuffix: str = '',
        indexFilter: ScanIndexFilter = ScanIndexFilter.ALL,
    ) -> int:
        reconstructor = self._reconstructorChooser.get_current_plugin().strategy
        parameters = self._dataMatcher.matchDiffractionPatternsWithPositions(
            inputProductIndex, indexFilter
        )
        outputProductIndex = self._productRepository.insertNewProduct(likeIndex=inputProductIndex)
        outputProduct = self._productRepository[outputProductIndex]

        outputProductName = (
            self._dataMatcher.getProductName(inputProductIndex)
            + f'_{self._reconstructorChooser.get_current_plugin().simple_name}'
        )

        if outputProductSuffix:
            outputProductName += f'_{outputProductSuffix}'

        outputProduct.setName(outputProductName)
        self._reconstructionQueue.put(reconstructor, parameters, outputProduct)
        return outputProductIndex

    def reconstructSplit(self, inputProductIndex: int) -> tuple[int, int]:
        outputProductIndexOdd = self.reconstruct(
            inputProductIndex,
            outputProductSuffix='odd',
            indexFilter=ScanIndexFilter.ODD,
        )
        outputProductIndexEven = self.reconstruct(
            inputProductIndex,
            outputProductSuffix='even',
            indexFilter=ScanIndexFilter.EVEN,
        )

        return outputProductIndexOdd, outputProductIndexEven

    def openModel(self, filePath: Path) -> None:
        reconstructor = self._reconstructorChooser.get_current_plugin().strategy

        if isinstance(reconstructor, TrainableReconstructor):
            logger.info('Opening model...')
            tic = time.perf_counter()
            reconstructor.openModel(filePath)
            toc = time.perf_counter()
            logger.info(f'Open time {toc - tic:.4f} seconds.')
        else:
            logger.warning('Reconstructor is not trainable!')

    def saveModel(self, filePath: Path) -> None:
        reconstructor = self._reconstructorChooser.get_current_plugin().strategy

        if isinstance(reconstructor, TrainableReconstructor):
            logger.info('Saving model...')
            tic = time.perf_counter()
            reconstructor.saveModel(filePath)
            toc = time.perf_counter()
            logger.info(f'Save time {toc - tic:.4f} seconds.')
        else:
            logger.warning('Reconstructor is not trainable!')

    def exportTrainingData(self, filePath: Path, inputProductIndex: int) -> None:
        reconstructor = self._reconstructorChooser.get_current_plugin().strategy

        if isinstance(reconstructor, TrainableReconstructor):
            logger.info('Preparing input data...')
            tic = time.perf_counter()
            parameters = self._dataMatcher.matchDiffractionPatternsWithPositions(
                inputProductIndex, ScanIndexFilter.ALL
            )
            toc = time.perf_counter()
            logger.info(f'Data preparation time {toc - tic:.4f} seconds.')

            logger.info('Exporting...')
            tic = time.perf_counter()
            reconstructor.exportTrainingData(filePath, parameters)
            toc = time.perf_counter()
            logger.info(f'Export time {toc - tic:.4f} seconds.')
        else:
            logger.warning('Reconstructor is not trainable!')

    def train(self, dataPath: Path) -> TrainOutput:
        reconstructor = self._reconstructorChooser.get_current_plugin().strategy
        result = TrainOutput([], [], -1)

        if isinstance(reconstructor, TrainableReconstructor):
            logger.info('Training...')
            tic = time.perf_counter()
            result = reconstructor.train(dataPath)
            toc = time.perf_counter()
            logger.info(f'Training time {toc - tic:.4f} seconds. (code={result.result})')
        else:
            logger.warning('Reconstructor is not trainable!')

        return result

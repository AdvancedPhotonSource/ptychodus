from collections.abc import Iterator, Sequence
from pathlib import Path
import logging

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.plugins import PluginChooser
from ptychodus.api.reconstructor import (
    Reconstructor,
    TrainableReconstructor,
    TrainOutput,
)

from .api import ReconstructorAPI
from .log import ReconstructorLogHandler
from .settings import ReconstructorSettings

logger = logging.getLogger(__name__)


class ReconstructorPresenter(Observable, Observer):
    def __init__(
        self,
        settings: ReconstructorSettings,
        reconstructorChooser: PluginChooser[Reconstructor],
        logHandler: ReconstructorLogHandler,
        reconstructorAPI: ReconstructorAPI,
        reinitObservable: Observable,
    ) -> None:
        super().__init__()
        self._settings = settings
        self._reconstructorChooser = reconstructorChooser
        self._logHandler = logHandler
        self._reconstructorAPI = reconstructorAPI
        self._reinitObservable = reinitObservable

        reconstructorChooser.addObserver(self)
        reinitObservable.addObserver(self)
        self._syncFromSettings()

    def getReconstructorList(self) -> Sequence[str]:
        return self._reconstructorChooser.getDisplayNameList()

    def getReconstructor(self) -> str:
        return self._reconstructorChooser.currentPlugin.displayName

    def setReconstructor(self, name: str) -> None:
        self._reconstructorChooser.setCurrentPluginByName(name)

    def _syncFromSettings(self) -> None:
        self.setReconstructor(self._settings.algorithm.getValue())

    def _syncToSettings(self) -> None:
        self._settings.algorithm.setValue(self._reconstructorChooser.currentPlugin.simpleName)

    def reconstruct(self, inputProductIndex: int) -> int:
        return self._reconstructorAPI.reconstruct(inputProductIndex)

    def reconstructSplit(self, inputProductIndex: int) -> tuple[int, int]:
        return self._reconstructorAPI.reconstructSplit(inputProductIndex)

    @property
    def isReconstructing(self) -> bool:
        return self._reconstructorAPI.isReconstructing

    def flushLog(self) -> Iterator[str]:
        for text in self._logHandler.messages():
            yield text

    def processResults(self, *, block: bool) -> None:
        self._reconstructorAPI.processResults(block=block)

    @property
    def isTrainable(self) -> bool:
        reconstructor = self._reconstructorChooser.currentPlugin.strategy
        return isinstance(reconstructor, TrainableReconstructor)

    def getModelFileFilter(self) -> str:
        reconstructor = self._reconstructorChooser.currentPlugin.strategy

        if isinstance(reconstructor, TrainableReconstructor):
            return reconstructor.getModelFileFilter()
        else:
            logger.warning('Reconstructor is not trainable!')

        return str()

    def openModel(self, filePath: Path) -> None:
        return self._reconstructorAPI.openModel(filePath)

    def saveModel(self, filePath: Path) -> None:
        return self._reconstructorAPI.saveModel(filePath)

    def getTrainingDataFileFilter(self) -> str:
        reconstructor = self._reconstructorChooser.currentPlugin.strategy

        if isinstance(reconstructor, TrainableReconstructor):
            return reconstructor.getTrainingDataFileFilter()
        else:
            logger.warning('Reconstructor is not trainable!')

        return str()

    def exportTrainingData(self, filePath: Path, inputProductIndex: int) -> None:
        return self._reconstructorAPI.exportTrainingData(filePath, inputProductIndex)

    def train(self, dataPath: Path) -> TrainOutput:
        return self._reconstructorAPI.train(dataPath)

    def update(self, observable: Observable) -> None:
        if observable is self._reconstructorChooser:
            self._syncToSettings()
            self.notifyObservers()
        elif observable is self._reinitObservable:
            self._syncFromSettings()

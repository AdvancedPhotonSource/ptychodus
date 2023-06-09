from __future__ import annotations
from collections.abc import Sequence
from pathlib import Path
import logging

import numpy

from ...api.observer import Observable, Observer
from .active import ActiveDiffractionDataset
from .api import DiffractionDataAPI
from .settings import DiffractionDatasetSettings

logger = logging.getLogger(__name__)


class DiffractionDatasetInputOutputPresenter(Observable, Observer):

    def __init__(self, settings: DiffractionDatasetSettings, dataset: ActiveDiffractionDataset,
                 dataAPI: DiffractionDataAPI, reinitObservable: Observable) -> None:
        super().__init__()
        self._settings = settings
        self._dataset = dataset
        self._dataAPI = dataAPI
        self._reinitObservable = reinitObservable

    @classmethod
    def createInstance(cls, settings: DiffractionDatasetSettings,
                       dataset: ActiveDiffractionDataset, dataAPI: DiffractionDataAPI,
                       reinitObservable: Observable) -> DiffractionDatasetInputOutputPresenter:
        presenter = cls(settings, dataset, dataAPI, reinitObservable)
        reinitObservable.addObserver(presenter)
        return presenter

    def getOpenFileFilterList(self) -> Sequence[str]:
        return self._dataAPI.getOpenFileFilterList()

    def getOpenFileFilter(self) -> str:
        return self._dataAPI.getOpenFileFilter()

    def openDiffractionFile(self, filePath: Path, fileFilter: str) -> None:
        try:
            fileType = self._dataAPI.loadDiffractionDataset(filePath=filePath,
                                                            displayFileType=fileFilter,
                                                            assemble=False)
        except Exception:
            logger.exception('Failed to load diffraction dataset.')
            return

        if fileType is None:
            logger.error('Failed to load diffraction dataset.')
        else:
            self._settings.fileType.value = fileType
            self._settings.filePath.value = filePath

        self.notifyObservers()

    def _openDiffractionFileFromSettings(self) -> None:
        self._dataAPI.loadDiffractionDataset(filePath=self._settings.filePath.value,
                                             simpleFileType=self._settings.fileType.value,
                                             assemble=True)

    def startAssemblingDiffractionPatterns(self) -> None:
        self._dataAPI.startAssemblingDiffractionPatterns()

    def stopAssemblingDiffractionPatterns(self, finishAssembling: bool) -> None:
        self._dataAPI.stopAssemblingDiffractionPatterns(finishAssembling)

    def getSaveFileFilterList(self) -> Sequence[str]:
        return [self.getSaveFileFilter()]

    def getSaveFileFilter(self) -> str:
        return 'NumPy Binary Files (*.npy)'

    def saveDiffractionFile(self, filePath: Path) -> None:
        # TODO saveDiffractionFile should share code with state data I/O
        fileFilter = self.getSaveFileFilter()
        logger.debug(f'Writing \"{filePath}\" as \"{fileFilter}\"')
        array = self._dataset.getAssembledData()
        numpy.save(filePath, array)

    def update(self, observable: Observable) -> None:
        if observable is self._reinitObservable:
            self._openDiffractionFileFromSettings()

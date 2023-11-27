from __future__ import annotations
from collections.abc import Sequence
from pathlib import Path
from typing import overload
import logging

from ...api.experiment import Experiment, ExperimentFileReader, ExperimentFileWriter
from ...api.observer import Observable, ObservableSequence, Observer
from ...api.plugins import PluginChooser
from ...api.settings import SettingsRegistry
from .detector import Detector, DetectorPresenter, DetectorSettings

logger = logging.getLogger(__name__)


class ExperimentRepositoryPresenter(ObservableSequence[Experiment], Observer):

    def __init__(self, fileReaderChooser: PluginChooser[ExperimentFileReader],
                 fileWriterChooser: PluginChooser[ExperimentFileWriter]) -> None:
        super().__init__()
        self._fileReaderChooser = fileReaderChooser
        self._fileWriterChooser = fileWriterChooser
        self._experimentList: list[Experiment] = list()

    @classmethod
    def createInstance(
            cls, fileReaderChooser: PluginChooser[ExperimentFileReader],
            fileWriterChooser: PluginChooser[ExperimentFileWriter]
    ) -> ExperimentRepositoryPresenter:
        return cls(fileReaderChooser, fileWriterChooser)

    @overload
    def __getitem__(self, index: int) -> Experiment:
        ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[Experiment]:
        ...

    def __getitem__(self, index: int | slice) -> Experiment | Sequence[Experiment]:
        return self._experimentList[index]

    def __len__(self) -> int:
        return len(self._experimentList)

    def getOpenFileFilterList(self) -> Sequence[str]:
        return self._fileReaderChooser.getDisplayNameList()

    def getOpenFileFilter(self) -> str:
        return self._fileReaderChooser.currentPlugin.displayName

    def openExperiment(self, filePath: Path, fileFilter: str) -> None:
        pass  # FIXME

    def getSaveFileFilterList(self) -> Sequence[str]:
        return self._fileWriterChooser.getDisplayNameList()

    def getSaveFileFilter(self) -> str:
        return self._fileWriterChooser.currentPlugin.displayName

    def saveExperiment(self, index: int, filePath: Path, fileFilter: str) -> None:
        try:
            experiment = self._experimentList[index]
        except IndexError:
            logger.debug(f'Failed to save experiment {index}!')
            return

        self._fileWriterChooser.setCurrentPluginByName(fileFilter)
        fileType = self._fileWriterChooser.currentPlugin.simpleName
        logger.debug(f'Writing \"{filePath}\" as \"{fileType}\"')
        writer = self._fileWriterChooser.currentPlugin.strategy
        writer.write(filePath, experiment)

    def insertExperiment(self, name: str) -> None:
        existingNames = {experiment.getName() for experiment in self._experimentList}
        uniqueName = name
        index = 0

        while uniqueName in existingNames:
            index += 1
            uniqueName = f'{name}-{index}'

        experiment = Experiment(uniqueName)
        experiment.addObserver(self)
        index = len(self._experimentList)
        self._experimentList.append(experiment)
        self.notifyObserversItemInserted(index)

    def removeExperiment(self, index: int) -> None:
        try:
            experiment = self._experimentList.pop(index)
        except IndexError:
            logger.debug(f'Failed to remove experiment {index}!')
        else:
            experiment.removeObserver(self)
            self.notifyObserversItemRemoved(index)

    def update(self, observable: Observable) -> None:
        if isinstance(observable, Experiment):
            try:
                index = self._experimentList.index(observable)
            except ValueError:
                logger.warning(f'Failed to locate experiment \"{observable}\"!')
            else:
                self.notifyObserversItemChanged(index)
        else:
            logger.warning(f'Observable is not an experiment \"{observable}\"!')


class ExperimentCore:

    def __init__(self, settingsRegistry: SettingsRegistry,
                 fileReaderChooser: PluginChooser[ExperimentFileReader],
                 fileWriterChooser: PluginChooser[ExperimentFileWriter]) -> None:
        self.detectorSettings = DetectorSettings.createInstance(settingsRegistry)
        self.detector = Detector.createInstance(self.detectorSettings)
        self.detectorPresenter = DetectorPresenter.createInstance(self.detectorSettings,
                                                                  self.detector)
        self.repositoryPresenter = ExperimentRepositoryPresenter.createInstance(
            fileReaderChooser, fileWriterChooser)

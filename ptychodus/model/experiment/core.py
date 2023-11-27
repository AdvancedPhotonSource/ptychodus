from __future__ import annotations
from collections.abc import Sequence
from pathlib import Path
from typing import overload
import logging

from ...api.experiment import Experiment, ExperimentFileReader, ExperimentFileWriter
from ...api.observer import Observable, ObservableSequence, Observer
from ...api.plugins import PluginChooser
from ...api.settings import SettingsRegistry
from .detector import Detector, DetectorPresenter

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

    def _generateUniqueName(self, nameHint: str) -> str:
        existingNames = {experiment.getName() for experiment in self._experimentList}
        uniqueName = nameHint
        index = 0

        while uniqueName in existingNames:
            index += 1
            uniqueName = f'{nameHint}-{index}'

        return uniqueName

    def _appendExperiment(self, experiment: Experiment) -> None:
        experiment.addObserver(self)
        index = len(self._experimentList)
        self._experimentList.append(experiment)
        self.notifyObserversItemInserted(index)

    def getOpenFileFilterList(self) -> Sequence[str]:
        return self._fileReaderChooser.getDisplayNameList()

    def getOpenFileFilter(self) -> str:
        return self._fileReaderChooser.currentPlugin.displayName

    def openExperiment(self, filePath: Path, fileFilter: str) -> None:
        if filePath.is_file():
            self._fileReaderChooser.setCurrentPluginByName(fileFilter)
            fileType = self._fileReaderChooser.currentPlugin.simpleName
            logger.debug(f'Reading \"{filePath}\" as \"{fileType}\"')
            fileReader = self._fileReaderChooser.currentPlugin.strategy

            try:
                experiment = fileReader.read(filePath)
            except Exception as exc:
                raise RuntimeError(f'Failed to read \"{filePath}\"') from exc
            else:
                uniqueName = self._generateUniqueName(experiment.getName())
                experiment.setName(uniqueName)
                self._appendExperiment(experiment)
        else:
            logger.debug(f'Refusing to create experiment with invalid file path \"{filePath}\"')

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

    def insertExperiment(self, nameHint: str = 'Unnamed') -> None:
        uniqueName = self._generateUniqueName(nameHint)
        experiment = Experiment(uniqueName)
        self._appendExperiment(experiment)

    def removeExperiment(self, index: int) -> None:
        try:
            experiment = self._experimentList.pop(index)
        except IndexError:
            logger.debug(f'Failed to remove experiment {index}!')
        else:
            experiment.removeObserver(self)
            self.notifyObserversItemRemoved(index)

    def getInfoText(self) -> str:
        sizeInMB = sum(exp.getSizeInBytes() for exp in self._experimentList) / (1024 * 1024)
        return f'Total: {len(self)} [{sizeInMB:.2f}MB]'

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
        self.detector = Detector.createInstance(settingsRegistry)
        self.detectorPresenter = DetectorPresenter.createInstance(self.detector)
        self.repositoryPresenter = ExperimentRepositoryPresenter.createInstance(
            fileReaderChooser, fileWriterChooser)

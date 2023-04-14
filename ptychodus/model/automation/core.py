from __future__ import annotations
from pathlib import Path
from typing import Generator
import queue

from ...api.geometry import Interval
from ...api.observer import Observable, Observer
from ...api.settings import SettingsRegistry, SettingsGroup
from ..data import DataCore
from ..object import ObjectAPI
from ..probe import ProbeCore
from ..scan import ScanAPI
from ..workflow import WorkflowCore
from .buffer import AutomationDatasetBuffer
from .processor import AutomationDatasetProcessor
from .repository import AutomationDatasetRepository, AutomationDatasetState
from .settings import AutomationSettings
from .watcher import DataDirectoryWatcher
from .workflow import S02AutomationDatasetWorkflow


class AutomationPresenter(Observable, Observer):

    def __init__(self, settings: AutomationSettings, watcher: DataDirectoryWatcher,
                 datasetBuffer: AutomationDatasetBuffer) -> None:
        super().__init__()
        self._settings = settings
        self._watcher = watcher
        self._datasetBuffer = datasetBuffer

    @classmethod
    def createInstance(cls, settings: AutomationSettings, watcher: DataDirectoryWatcher,
                       datasetBuffer: AutomationDatasetBuffer) -> AutomationPresenter:
        presenter = cls(settings, watcher, datasetBuffer)
        settings.addObserver(presenter)
        watcher.addObserver(presenter)
        return presenter

    def getStrategyList(self) -> list[str]:
        return [
            'LYNX Catalyst Particle',
            'CNM/APS Hard X-Ray Nanoprobe',
        ]

    def getStrategy(self) -> str:
        return self._settings.strategy.value

    def setStrategy(self, strategy: str) -> None:
        self._settings.strategy.value = strategy

    def getDataDirectory(self) -> Path:
        return self._settings.dataDirectory.value

    def setDataDirectory(self, directory: Path) -> None:
        self._settings.dataDirectory.value = directory

    def getProcessingIntervalLimitsInSeconds(self) -> Interval[int]:
        return Interval[int](0, 600)

    def getProcessingIntervalInSeconds(self) -> int:
        limits = self.getProcessingIntervalLimitsInSeconds()
        return limits.clamp(self._settings.processingIntervalInSeconds.value)

    def setProcessingIntervalInSeconds(self, value: int) -> None:
        self._settings.processingIntervalInSeconds.value = value

    def execute(self) -> None:  # TODO generalize
        dataDirectory = self.getDataDirectory()
        scanFileGlob: Generator[Path, None, None] = dataDirectory.glob('*.csv')

        for scanFile in scanFileGlob:
            self._datasetBuffer.put(scanFile)

    def isWatchdogEnabled(self) -> bool:
        return self._watcher.isAlive

    def setWatchdogEnabled(self, enable: bool) -> None:
        if enable:
            self._watcher.start()
        else:
            self._watcher.stop()

    def getWatchdogDelayLimitsInSeconds(self) -> Interval[int]:
        return Interval[int](0, 600)

    def getWatchdogDelayInSeconds(self) -> int:
        limits = self.getWatchdogDelayLimitsInSeconds()
        return limits.clamp(self._settings.watchdogDelayInSeconds.value)

    def setWatchdogDelayInSeconds(self, value: int) -> None:
        self._settings.watchdogDelayInSeconds.value = value

    def setWatchdogPollingObserverEnabled(self, enable: bool) -> None:
        self._settings.useWatchdogPollingObserver.value = enable

    def isWatchdogPollingObserverEnabled(self) -> bool:
        return self._settings.useWatchdogPollingObserver.value

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notifyObservers()
        elif observable is self._watcher:
            self.notifyObservers()


class AutomationProcessingPresenter(Observable, Observer):

    def __init__(self, settings: AutomationSettings, repository: AutomationDatasetRepository,
                 processor: AutomationDatasetProcessor) -> None:
        super().__init__()
        self._settings = settings
        self._repository = repository
        self._processor = processor

    @classmethod
    def createInstance(cls, settings: AutomationSettings, repository: AutomationDatasetRepository,
                       processor: AutomationDatasetProcessor) -> AutomationProcessingPresenter:
        presenter = cls(settings, repository, processor)
        settings.addObserver(presenter)
        repository.addObserver(presenter)
        return presenter

    def getDatasetLabel(self, index: int) -> str:
        return self._repository.getLabel(index)

    def getDatasetState(self, index: int) -> AutomationDatasetState:
        return self._repository.getState(index)

    def getNumberOfDatasets(self) -> int:
        return len(self._repository)

    def isProcessingEnabled(self) -> bool:
        return self._processor.isAlive

    def setProcessingEnabled(self, enable: bool) -> None:
        if enable:
            self._processor.start()
        else:
            self._processor.stop()

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notifyObservers()
        elif observable is self._repository:
            self.notifyObservers()


class AutomationCore:

    def __init__(self, settingsRegistry: SettingsRegistry, dataCore: DataCore, scanAPI: ScanAPI,
                 probeCore: ProbeCore, objectAPI: ObjectAPI, workflowCore: WorkflowCore) -> None:
        self._settings = AutomationSettings.createInstance(settingsRegistry)
        self.repository = AutomationDatasetRepository(self._settings)
        self._workflow = S02AutomationDatasetWorkflow(dataCore, scanAPI, probeCore, objectAPI,
                                                      workflowCore)
        self._processingQueue: queue.Queue[Path] = queue.Queue()
        self._processor = AutomationDatasetProcessor(self._settings, self.repository,
                                                     self._workflow, self._processingQueue)
        self._datasetBuffer = AutomationDatasetBuffer(self._settings, self.repository,
                                                      self._processor)
        self._watcher = DataDirectoryWatcher.createInstance(self._settings, self._datasetBuffer)
        self.presenter = AutomationPresenter.createInstance(self._settings, self._watcher,
                                                            self._datasetBuffer)
        self.processingPresenter = AutomationProcessingPresenter.createInstance(
            self._settings, self.repository, self._processor)

    def start(self) -> None:
        self._datasetBuffer.start()

    def executeWaitingTasks(self) -> None:
        # TODO this belongs in AutomationDatasetProcessor
        try:
            filePath = self._processingQueue.get(block=False)

            try:
                self.repository.put(filePath, AutomationDatasetState.PROCESSING)
                self._workflow.execute(filePath)
                self.repository.put(filePath, AutomationDatasetState.COMPLETE)
            finally:
                self._processingQueue.task_done()
        except queue.Empty:
            pass

    def stop(self) -> None:
        self._processor.stop()
        self._watcher.stop()
        self._datasetBuffer.stop()

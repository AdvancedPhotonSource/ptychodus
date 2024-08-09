from __future__ import annotations
from collections.abc import Sequence
from pathlib import Path
import queue

from ptychodus.api.geometry import Interval
from ptychodus.api.observer import Observable, Observer
from ptychodus.api.plugins import PluginChooser
from ptychodus.api.settings import SettingsRegistry
from ptychodus.api.workflow import FileBasedWorkflow, WorkflowAPI

from .buffer import AutomationDatasetBuffer
from .processor import AutomationDatasetProcessor
from .repository import AutomationDatasetRepository, AutomationDatasetState
from .settings import AutomationSettings
from .watcher import DataDirectoryWatcher
from .workflow import CurrentFileBasedWorkflow


class AutomationPresenter(Observable, Observer):

    def __init__(self, settings: AutomationSettings, workflow: CurrentFileBasedWorkflow,
                 watcher: DataDirectoryWatcher, datasetBuffer: AutomationDatasetBuffer,
                 datasetRepository: AutomationDatasetRepository) -> None:
        super().__init__()
        self._settings = settings
        self._workflow = workflow
        self._watcher = watcher
        self._datasetBuffer = datasetBuffer
        self._datasetRepository = datasetRepository

        settings.addObserver(self)
        watcher.addObserver(self)

    def getStrategyList(self) -> Sequence[str]:
        return self._workflow.getAvailableWorkflows()

    def getStrategy(self) -> str:
        return self._workflow.getWorkflow()

    def setStrategy(self, strategy: str) -> None:
        self._workflow.setWorkflow(strategy)

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

    def loadExistingDatasetsToRepository(self) -> None:
        dataDirectory = self.getDataDirectory()
        pattern = '**/' if self._workflow.isWatchRecursive else ''
        pattern += self._workflow.getWatchFilePattern()
        scanFileList = sorted(scanFile for scanFile in dataDirectory.glob(pattern))

        for scanFile in scanFileList:
            self._datasetBuffer.put(scanFile)

    def clearDatasetRepository(self) -> None:
        self._datasetRepository.clear()

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

    def __init__(self, settingsRegistry: SettingsRegistry, workflowAPI: WorkflowAPI,
                 workflowChooser: PluginChooser[FileBasedWorkflow]) -> None:
        self._settings = AutomationSettings(settingsRegistry)
        self.repository = AutomationDatasetRepository(self._settings)
        self._workflow = CurrentFileBasedWorkflow(self._settings, workflowChooser)
        self._processingQueue: queue.Queue[Path] = queue.Queue()
        self._processor = AutomationDatasetProcessor(self._settings, self.repository,
                                                     self._workflow, workflowAPI,
                                                     self._processingQueue)
        self._datasetBuffer = AutomationDatasetBuffer(self._settings, self.repository,
                                                      self._processor)
        self._watcher = DataDirectoryWatcher(self._settings, self._workflow, self._datasetBuffer)
        self.presenter = AutomationPresenter(self._settings, self._workflow, self._watcher,
                                             self._datasetBuffer, self.repository)
        self.processingPresenter = AutomationProcessingPresenter.createInstance(
            self._settings, self.repository, self._processor)

    def start(self) -> None:
        self._datasetBuffer.start()

    def executeWaitingTasks(self) -> None:
        self._processor.runOnce()

    def stop(self) -> None:
        self._processor.stop()
        self._watcher.stop()
        self._datasetBuffer.stop()

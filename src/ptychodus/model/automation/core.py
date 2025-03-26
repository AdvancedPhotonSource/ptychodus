from __future__ import annotations
from collections.abc import Iterator
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
    def __init__(
        self,
        settings: AutomationSettings,
        workflow: CurrentFileBasedWorkflow,
        watcher: DataDirectoryWatcher,
        datasetBuffer: AutomationDatasetBuffer,
        datasetRepository: AutomationDatasetRepository,
    ) -> None:
        super().__init__()
        self._settings = settings
        self._workflow = workflow
        self._watcher = watcher
        self._datasetBuffer = datasetBuffer
        self._datasetRepository = datasetRepository

        settings.add_observer(self)
        watcher.add_observer(self)

    def get_strategies(self) -> Iterator[str]:
        return self._workflow.getAvailableWorkflows()

    def get_strategy(self) -> str:
        return self._workflow.getWorkflow()

    def set_strategy(self, strategy: str) -> None:
        self._workflow.setWorkflow(strategy)

    def getDataDirectory(self) -> Path:
        return self._settings.dataDirectory.get_value()

    def set_data_directory(self, directory: Path) -> None:
        self._settings.dataDirectory.set_value(directory)

    def get_processing_interval_limits_s(self) -> Interval[int]:
        return Interval[int](0, 600)

    def get_processing_interval_s(self) -> int:
        limits = self.get_processing_interval_limits_s()
        return limits.clamp(self._settings.processingIntervalInSeconds.get_value())

    def set_processing_interval_s(self, value: int) -> None:
        self._settings.processingIntervalInSeconds.set_value(value)

    def loadExistingDatasetsToRepository(self) -> None:
        dataDirectory = self.getDataDirectory()
        pattern = '**/' if self._workflow.is_watch_recursive else ''
        pattern += self._workflow.get_watch_file_pattern()
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

    def get_watchdog_delay_limits_s(self) -> Interval[int]:
        return Interval[int](0, 600)

    def get_watchdog_delay_s(self) -> int:
        limits = self.get_watchdog_delay_limits_s()
        return limits.clamp(self._settings.watchdogDelayInSeconds.get_value())

    def setWatchdogDelayInSeconds(self, value: int) -> None:
        self._settings.watchdogDelayInSeconds.set_value(value)

    def setWatchdogPollingObserverEnabled(self, enable: bool) -> None:
        self._settings.useWatchdogPollingObserver.set_value(enable)

    def is_watchdog_polling_observer_enabled(self) -> bool:
        return self._settings.useWatchdogPollingObserver.get_value()

    def _update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notify_observers()
        elif observable is self._watcher:
            self.notify_observers()


class AutomationProcessingPresenter(Observable, Observer):
    def __init__(
        self,
        settings: AutomationSettings,
        repository: AutomationDatasetRepository,
        processor: AutomationDatasetProcessor,
    ) -> None:
        super().__init__()
        self._settings = settings
        self._repository = repository
        self._processor = processor

        settings.add_observer(self)
        repository.add_observer(self)

    def get_dataset_label(self, index: int) -> str:
        return self._repository.getLabel(index)

    def get_dataset_state(self, index: int) -> AutomationDatasetState:
        return self._repository.getState(index)

    def get_num_datasets(self) -> int:
        return len(self._repository)

    def isProcessingEnabled(self) -> bool:
        return self._processor.isAlive

    def setProcessingEnabled(self, enable: bool) -> None:
        if enable:
            self._processor.start()
        else:
            self._processor.stop()

    def _update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notify_observers()
        elif observable is self._repository:
            self.notify_observers()


class AutomationCore:
    def __init__(
        self,
        settingsRegistry: SettingsRegistry,
        workflowAPI: WorkflowAPI,
        workflowChooser: PluginChooser[FileBasedWorkflow],
    ) -> None:
        self._settings = AutomationSettings(settingsRegistry)
        self.repository = AutomationDatasetRepository(self._settings)
        self._workflow = CurrentFileBasedWorkflow(self._settings, workflowChooser)
        self._processingQueue: queue.Queue[Path] = queue.Queue()
        self._processor = AutomationDatasetProcessor(
            self._settings,
            self.repository,
            self._workflow,
            workflowAPI,
            self._processingQueue,
        )
        self._datasetBuffer = AutomationDatasetBuffer(
            self._settings, self.repository, self._processor
        )
        self._watcher = DataDirectoryWatcher(self._settings, self._workflow, self._datasetBuffer)
        self.presenter = AutomationPresenter(
            self._settings,
            self._workflow,
            self._watcher,
            self._datasetBuffer,
            self.repository,
        )
        self.processing_presenter = AutomationProcessingPresenter(
            self._settings, self.repository, self._processor
        )

    def start(self) -> None:
        self._datasetBuffer.start()

    def refreshDatasetRepository(self) -> None:
        self.repository.notifyObserversIfRepositoryChanged()

    def executeWaitingTasks(self) -> None:
        self._processor.runOnce()

    def stop(self) -> None:
        self._processor.stop()
        self._watcher.stop()
        self._datasetBuffer.stop()

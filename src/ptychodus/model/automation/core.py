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
        dataset_buffer: AutomationDatasetBuffer,
        dataset_repository: AutomationDatasetRepository,
    ) -> None:
        super().__init__()
        self._settings = settings
        self._workflow = workflow
        self._watcher = watcher
        self._dataset_buffer = dataset_buffer
        self._dataset_repository = dataset_repository

        settings.add_observer(self)
        watcher.add_observer(self)

    def get_strategies(self) -> Iterator[str]:
        return self._workflow.get_available_workflows()

    def get_strategy(self) -> str:
        return self._workflow.get_workflow()

    def set_strategy(self, strategy: str) -> None:
        self._workflow.set_workflow(strategy)

    def get_data_directory(self) -> Path:
        return self._settings.data_directory.get_value()

    def set_data_directory(self, directory: Path) -> None:
        self._settings.data_directory.set_value(directory)

    def get_processing_interval_limits_s(self) -> Interval[int]:
        return Interval[int](0, 600)

    def get_processing_interval_s(self) -> int:
        limits = self.get_processing_interval_limits_s()
        return limits.clamp(self._settings.processing_interval_s.get_value())

    def set_processing_interval_s(self, value: int) -> None:
        self._settings.processing_interval_s.set_value(value)

    def load_existing_datasets_to_repository(self) -> None:
        data_directory = self.get_data_directory()
        pattern = '**/' if self._workflow.is_watch_recursive else ''
        pattern += self._workflow.get_watch_file_pattern()
        scan_file_list = sorted(scanFile for scanFile in data_directory.glob(pattern))

        for scan_file in scan_file_list:
            self._dataset_buffer.put(scan_file)

    def clear_dataset_repository(self) -> None:
        self._dataset_repository.clear()

    def is_watchdog_enabled(self) -> bool:
        return self._watcher.is_alive

    def set_watchdog_enabled(self, enable: bool) -> None:
        if enable:
            self._watcher.start()
        else:
            self._watcher.stop()

    def get_watchdog_delay_limits_s(self) -> Interval[int]:
        return Interval[int](0, 600)

    def get_watchdog_delay_s(self) -> int:
        limits = self.get_watchdog_delay_limits_s()
        return limits.clamp(self._settings.watchdog_delay_s.get_value())

    def set_watchdog_delay_s(self, value: int) -> None:
        self._settings.watchdog_delay_s.set_value(value)

    def set_watchdog_polling_observer_enabled(self, enable: bool) -> None:
        self._settings.use_watchdog_polling_observer.set_value(enable)

    def is_watchdog_polling_observer_enabled(self) -> bool:
        return self._settings.use_watchdog_polling_observer.get_value()

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
        return self._repository.get_label(index)

    def get_dataset_state(self, index: int) -> AutomationDatasetState:
        return self._repository.get_state(index)

    def get_num_datasets(self) -> int:
        return len(self._repository)

    def is_processing_enabled(self) -> bool:
        return self._processor.is_alive

    def set_processing_enabled(self, enable: bool) -> None:
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
        settings_registry: SettingsRegistry,
        workflow_api: WorkflowAPI,
        workflow_chooser: PluginChooser[FileBasedWorkflow],
    ) -> None:
        self._settings = AutomationSettings(settings_registry)
        self.repository = AutomationDatasetRepository(self._settings)
        self._workflow = CurrentFileBasedWorkflow(self._settings, workflow_chooser)
        self._processing_queue: queue.Queue[Path] = queue.Queue()
        self._processor = AutomationDatasetProcessor(
            self._settings,
            self.repository,
            self._workflow,
            workflow_api,
            self._processing_queue,
        )
        self._dataset_buffer = AutomationDatasetBuffer(
            self._settings, self.repository, self._processor
        )
        self._watcher = DataDirectoryWatcher(self._settings, self._workflow, self._dataset_buffer)
        self.presenter = AutomationPresenter(
            self._settings,
            self._workflow,
            self._watcher,
            self._dataset_buffer,
            self.repository,
        )
        self.processing_presenter = AutomationProcessingPresenter(
            self._settings, self.repository, self._processor
        )

    def start(self) -> None:
        self._dataset_buffer.start()

    def refresh_dataset_repository(self) -> None:
        self.repository.notify_observers_if_repository_changed()

    def execute_waiting_tasks(self) -> None:
        self._processor.run_once()

    def stop(self) -> None:
        self._processor.stop()
        self._watcher.stop()
        self._dataset_buffer.stop()

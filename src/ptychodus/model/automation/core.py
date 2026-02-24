from __future__ import annotations
from collections.abc import Iterator

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.plugins import Plugin, PluginChooser
from ptychodus.api.settings import SettingsRegistry
from ptychodus.api.workflow import FileBasedWorkflow, WorkflowAPI

from ..task_manager import TaskManager
from .buffer import PendingFileBuffer
from .repository import AutomationDatasetRepository
from .settings import AutomationSettings
from .watcher import DataDirectoryWatcher


class AutomationPresenter(Observable, Observer):
    def __init__(
        self,
        settings: AutomationSettings,
        watcher: DataDirectoryWatcher,
        dataset_buffer: PendingFileBuffer,
        workflow_chooser: PluginChooser[FileBasedWorkflow],
    ) -> None:
        super().__init__()
        self._settings = settings
        self._watcher = watcher
        self._dataset_buffer = dataset_buffer
        self._workflow_chooser = workflow_chooser

        watcher.add_observer(self)

    @property
    def _current_workflow_plugin(self) -> Plugin[FileBasedWorkflow]:
        return self._workflow_chooser.get_current_plugin()

    @property
    def _current_workflow(self) -> FileBasedWorkflow:
        return self._current_workflow_plugin.strategy

    def available_workflows(self) -> Iterator[str]:
        for plugin in self._workflow_chooser:
            yield plugin.display_name

    def get_current_workflow(self) -> str:
        return self._current_workflow_plugin.display_name

    def set_current_workflow(self, name: str) -> None:
        self._workflow_chooser.set_current_plugin(name)

    def load_existing_datasets_to_repository(self) -> None:
        data_directory = self._settings.data_directory.get_value()
        pattern = '**/' if self._current_workflow_plugin.strategy.is_watch_recursive else ''
        pattern += self._current_workflow_plugin.strategy.get_watch_file_pattern()
        scan_file_list = sorted(scanFile for scanFile in data_directory.glob(pattern))

        for scan_file in scan_file_list:
            self._dataset_buffer.put(scan_file)

    def is_watcher_enabled(self) -> bool:
        return self._watcher.is_alive

    def set_watcher_enabled(self, enable: bool) -> None:
        if enable:
            self._watcher.start()
        else:
            self._watcher.stop()

    def _update(self, observable: Observable) -> None:
        if observable is self._watcher:
            self.notify_observers()


class AutomationCore:
    def __init__(
        self,
        task_manager: TaskManager,
        settings_registry: SettingsRegistry,
        workflow_api: WorkflowAPI,
        workflow_chooser: PluginChooser[FileBasedWorkflow],
    ) -> None:
        self.settings = AutomationSettings(settings_registry)
        self.repository = AutomationDatasetRepository(self.settings)
        self._file_buffer = PendingFileBuffer(
            task_manager, self.settings, self.repository, workflow_api, workflow_chooser
        )
        self._watcher = DataDirectoryWatcher(self.settings, self._file_buffer)
        self.presenter = AutomationPresenter(
            self.settings, self._watcher, self._file_buffer, workflow_chooser
        )

    def start(self) -> None:
        self._file_buffer.start()

    def stop(self) -> None:
        self._watcher.stop()
        self._file_buffer.stop()

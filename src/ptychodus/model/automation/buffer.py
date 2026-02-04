from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from time import monotonic as time
from typing import Final
import logging
import threading

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.plugins import PluginChooser
from ptychodus.api.workflow import FileBasedWorkflow, WorkflowAPI

from ..task_manager import TaskManager
from .repository import (
    AutomationDatasetRepository,
    AutomationDatasetState,
    UpdateAutomationDatasetRepository,
)
from .settings import AutomationSettings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FileStatus:
    stat_time_s: float
    mtime_ns: int
    size_bytes: int

    @classmethod
    def from_file(cls, file_path: Path) -> FileStatus:
        stat = file_path.stat()
        return cls(time(), stat.st_mtime_ns, stat.st_size)


class ExecuteWorkflow:
    def __init__(
        self,
        repository: AutomationDatasetRepository,
        workflow: FileBasedWorkflow,
        workflow_api: WorkflowAPI,
        file_path: Path,
    ) -> None:
        self._repository = repository
        self._workflow = workflow
        self._workflow_api = workflow_api
        self._file_path = file_path

    def __call__(self) -> None:
        self._repository.upsert(self._file_path, AutomationDatasetState.RUNNING)
        self._workflow.execute(self._workflow_api, self._file_path)
        self._repository.upsert(self._file_path, AutomationDatasetState.COMPLETE)


class PendingFileBuffer(Observable, Observer):
    WAIT_TIME_S: Final[int] = 5

    def __init__(
        self,
        task_manager: TaskManager,
        settings: AutomationSettings,
        repository: AutomationDatasetRepository,
        workflow_api: WorkflowAPI,
        workflow_chooser: PluginChooser[FileBasedWorkflow],
    ) -> None:
        super().__init__()
        self._task_manager = task_manager
        self._settings = settings
        self._repository = repository
        self._workflow_api = workflow_api
        self._workflow_chooser = workflow_chooser

        self._event_times_lock = threading.Lock()
        self._event_times: OrderedDict[Path, FileStatus] = OrderedDict()
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None

        self._watchdog_delay_lock = threading.Lock()
        self._watchdog_delay_s = settings.watchdog_delay_s.get_value()

        self._current_workflow_plugin_lock = threading.Lock()
        self._current_workflow_plugin = workflow_chooser.get_current_plugin()

        settings.watchdog_delay_s.add_observer(self)
        workflow_chooser.add_observer(self)

    def get_current_workflow(self) -> FileBasedWorkflow:
        with self._current_workflow_plugin_lock:
            return self._current_workflow_plugin.strategy

    def put(self, file_path: Path) -> None:
        file_status = FileStatus.from_file(file_path)

        with self._event_times_lock:
            self._event_times[file_path] = file_status
            self._event_times.move_to_end(file_path)

        task = UpdateAutomationDatasetRepository(
            self._repository, file_path, AutomationDatasetState.FOUND
        )
        self._task_manager.put_foreground_task(task)

    def _process_finalized_files(self) -> None:
        while not self._stop_event.is_set():
            try:
                with self._event_times_lock:
                    file_path, last_file_status = self._event_times.popitem(last=False)
            except KeyError:
                pass
            else:
                elapsed_time_s = time() - last_file_status.stat_time_s

                with self._watchdog_delay_lock:
                    remaining_time_s = self._watchdog_delay_s - elapsed_time_s

                if remaining_time_s > 0.0:
                    self._stop_event.wait(timeout=remaining_time_s)

                file_status = FileStatus.from_file(file_path)
                same_size = file_status.size_bytes == last_file_status.size_bytes
                same_mtime = file_status.mtime_ns == last_file_status.mtime_ns

                if same_size and same_mtime:
                    update_repository_task = UpdateAutomationDatasetRepository(
                        self._repository, file_path, AutomationDatasetState.WAITING
                    )
                    self._task_manager.put_foreground_task(update_repository_task)

                    with self._current_workflow_plugin_lock:
                        current_workflow = self._current_workflow_plugin.strategy

                    execute_workflow_task = ExecuteWorkflow(
                        self._repository, current_workflow, self._workflow_api, file_path
                    )
                    self._task_manager.put_foreground_task(execute_workflow_task)

                continue

            self._stop_event.wait(timeout=PendingFileBuffer.WAIT_TIME_S)

    def start(self) -> None:
        if self._worker is None:
            logger.info('Starting automation buffer...')
            self._stop_event.clear()
            self._worker = threading.Thread(target=self._process_finalized_files)
            self._worker.start()
            logger.info('Automation buffer started.')
        else:
            logger.info('Automation buffer already started!')

    def stop(self) -> None:
        if self._stop_event.is_set():
            logger.info('Automation buffer already stopped.')
        elif self._worker is None:
            logger.warning('Worker is None!')
        else:
            logger.info('Stopping automation buffer...')
            self._stop_event.set()
            self._worker.join()
            self._worker = None
            logger.info('Automation buffer stopped.')

    def _update(self, observable: Observable) -> None:
        if observable is self._settings.watchdog_delay_s:
            with self._watchdog_delay_lock:
                self._watchdog_delay_s = self._settings.watchdog_delay_s.get_value()
        if observable is self._workflow_chooser:
            with self._current_workflow_plugin_lock:
                self._current_workflow_plugin = self._workflow_chooser.get_current_plugin()

            self.notify_observers()

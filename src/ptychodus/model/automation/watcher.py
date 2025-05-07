from __future__ import annotations
from pathlib import Path
import logging

import watchdog.events
from watchdog.observers.polling import PollingObserver
import watchdog.observers

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.workflow import FileBasedWorkflow

from .buffer import AutomationDatasetBuffer
from .settings import AutomationSettings
from .workflow import CurrentFileBasedWorkflow

logger = logging.getLogger(__name__)


class DataDirectoryEventHandler(watchdog.events.FileSystemEventHandler):
    def __init__(
        self, workflow: FileBasedWorkflow, dataset_buffer: AutomationDatasetBuffer
    ) -> None:
        super().__init__()
        self._workflow = workflow
        self._dataset_buffer = dataset_buffer

    def on_created_or_modified(self, event: watchdog.events.FileSystemEvent) -> None:
        if not event.is_directory:
            src_path = Path(str(event.src_path))

            if src_path.match(self._workflow.get_watch_file_pattern()):
                self._dataset_buffer.put(src_path)

    def on_created(self, event: watchdog.events.FileSystemEvent) -> None:
        self.on_created_or_modified(event)

    def on_modified(self, event: watchdog.events.FileSystemEvent) -> None:
        self.on_created_or_modified(event)


class DataDirectoryWatcher(Observable, Observer):
    def __init__(
        self,
        settings: AutomationSettings,
        workflow: CurrentFileBasedWorkflow,
        dataset_buffer: AutomationDatasetBuffer,
    ) -> None:
        super().__init__()
        self._settings = settings
        self._workflow = workflow
        self._dataset_buffer = dataset_buffer
        self._observer: watchdog.observers.api.BaseObserver = watchdog.observers.Observer()

        settings.add_observer(self)
        workflow.add_observer(self)

    @property
    def is_alive(self) -> bool:
        return self._observer.is_alive()

    def _update_watch(self) -> None:
        self._observer.unschedule_all()
        data_directory = self._settings.data_directory.get_value()

        if data_directory.exists():
            observed_watch = self._observer.schedule(
                event_handler=DataDirectoryEventHandler(self._workflow, self._dataset_buffer),
                path=str(data_directory),
                recursive=self._workflow.is_watch_recursive,
            )
            logger.debug(observed_watch)
        else:
            logger.warning(f'Data directory "{data_directory}" does not exist!')

    def start(self) -> None:
        if self.is_alive:
            logger.error('Automation watchdog thread already started!')
        else:
            logger.info('Starting automation watchdog thread...')
            self._observer = (
                PollingObserver()
                if self._settings.use_watchdog_polling_observer.get_value()
                else watchdog.observers.Observer()
            )
            self._observer.start()
            self._update_watch()
            logger.debug('Automation watchdog thread started.')

    def stop(self) -> None:
        if self.is_alive:
            logger.info('Stopping automation watchdog thread...')
            self._observer.stop()
            self._observer.join()
            logger.debug('Automation watchdog thread stopped.')

    def _update(self, observable: Observable) -> None:
        if observable is self._settings:
            self._update_watch()
        elif observable is self._workflow:
            self._update_watch()

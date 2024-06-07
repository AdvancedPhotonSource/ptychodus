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

    def __init__(self, workflow: FileBasedWorkflow,
                 datasetBuffer: AutomationDatasetBuffer) -> None:
        super().__init__()
        self._workflow = workflow
        self._datasetBuffer = datasetBuffer

    def on_created_or_modified(self, event: watchdog.events.FileSystemEvent) -> None:
        if not event.is_directory:
            srcPath = Path(event.src_path)

            if srcPath.match(self._workflow.getFilePattern()):
                self._datasetBuffer.put(srcPath)

    def on_created(self, event: watchdog.events.FileSystemEvent) -> None:
        self.on_created_or_modified(event)

    def on_modified(self, event: watchdog.events.FileSystemEvent) -> None:
        self.on_created_or_modified(event)


class DataDirectoryWatcher(Observable, Observer):

    def __init__(self, settings: AutomationSettings, workflow: CurrentFileBasedWorkflow,
                 datasetBuffer: AutomationDatasetBuffer) -> None:
        super().__init__()
        self._settings = settings
        self._workflow = workflow
        self._datasetBuffer = datasetBuffer
        self._observer: watchdog.observers.api.BaseObserver = watchdog.observers.Observer()

        settings.addObserver(self)
        workflow.addObserver(self)

    @property
    def isAlive(self) -> bool:
        return self._observer.is_alive()

    def _updateWatch(self) -> None:
        self._observer.unschedule_all()
        dataDirectory = self._settings.dataDirectory.value

        if dataDirectory.exists():
            observedWatch = self._observer.schedule(
                event_handler=DataDirectoryEventHandler(self._workflow, self._datasetBuffer),
                path=dataDirectory,
                recursive=False,  # TODO generalize
            )
            logger.debug(observedWatch)
        else:
            logger.warning(f'Data directory \"{dataDirectory}\" does not exist!')

    def start(self) -> None:
        if self.isAlive:
            logger.error('Automation watchdog thread already started!')
        else:
            logger.info('Starting automation watchdog thread...')
            self._observer = PollingObserver() if self._settings.useWatchdogPollingObserver.value \
                    else watchdog.observers.Observer()
            self._observer.start()
            self._updateWatch()
            logger.debug('Automation watchdog thread started.')

    def stop(self) -> None:
        if self.isAlive:
            logger.info('Stopping automation watchdog thread...')
            self._observer.stop()
            self._observer.join()
            logger.debug('Automation watchdog thread stopped.')

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self._updateWatch()
        elif observable is self._workflow:
            self._updateWatch()

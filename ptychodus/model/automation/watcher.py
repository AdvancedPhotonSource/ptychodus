from __future__ import annotations
from pathlib import Path
import logging

import watchdog.events
import watchdog.observers

from ...api.observer import Observable, Observer
from .buffer import AutomationDatasetBuffer
from .settings import AutomationSettings

logger = logging.getLogger(__name__)


class DataDirectoryEventHandler(watchdog.events.FileSystemEventHandler):

    def __init__(self, datasetBuffer: AutomationDatasetBuffer) -> None:
        super().__init__()
        self._datasetBuffer = datasetBuffer

    def on_created_or_modified(self, event: watchdog.events.FileSystemEvent) -> None:
        if not event.is_directory:
            srcPath = Path(event.src_path)

            # TODO generalize
            if srcPath.suffix.casefold() == '.mda':
                self._datasetBuffer.put(srcPath)

    def on_created(self, event: watchdog.events.FileSystemEvent) -> None:
        self.on_created_or_modified(event)

    def on_modified(self, event: watchdog.events.FileSystemEvent) -> None:
        self.on_created_or_modified(event)


class DataDirectoryWatcher(Observable, Observer):

    def __init__(self, settings: AutomationSettings,
                 datasetBuffer: AutomationDatasetBuffer) -> None:
        super().__init__()
        self._settings = settings
        self._datasetBuffer = datasetBuffer
        self._observer = watchdog.observers.Observer()

    @classmethod
    def createInstance(cls, settings: AutomationSettings,
                       datasetBuffer: AutomationDatasetBuffer) -> DataDirectoryWatcher:
        watcher = cls(settings, datasetBuffer)
        settings.watchdogDirectory.addObserver(watcher)
        watcher._updateWatch()
        return watcher

    @property
    def isAlive(self) -> bool:
        return self._observer.is_alive()

    def start(self) -> None:
        if self.isAlive:
            logger.error('Automation watchdog thread already started!')
        else:
            logger.info('Starting automation watchdog thread...')
            self._observer = watchdog.observers.Observer()
            self._observer.start()
            observedWatch = self._observer.schedule(
                event_handler=DataDirectoryEventHandler(self._datasetBuffer),
                path=self._settings.watchdogDirectory.value,
                recursive=True,  # TODO generalize
            )
            logger.debug(observedWatch)
            logger.debug('Automation watchdog thread started.')

    def stop(self) -> None:
        if self.isAlive:
            logger.info('Stopping automation watchdog thread...')
            self._observer.stop()
            self._observer.join()
            logger.debug('Automation watchdog thread stopped.')

    def _updateWatch(self) -> None:
        logger.debug('Restart observer to update watch.')  # TODO

    def update(self, observable: Observable) -> None:
        if observable is self._settings.watchdogDirectory:
            self._updateWatch()

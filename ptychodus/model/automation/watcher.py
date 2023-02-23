from __future__ import annotations
from pathlib import Path
import logging
import queue
import threading
import watchdog.events
import watchdog.observers

from ...api.observer import Observable, Observer
from .settings import AutomationSettings

logger = logging.getLogger(__name__)


class DataDirectoryEventHandler(watchdog.events.FileSystemEventHandler):

    def __init__(self, dataDirectoryQueue: queue.Queue[Path]) -> None:
        super().__init__()
        self._dataDirectoryQueue = dataDirectoryQueue

    def on_created(self, event: watchdog.events.FileSystemEvent) -> None:
        srcPath = Path(event.src_path)

        if srcPath.is_dir():
            self._dataDirectoryQueue.put(srcPath)
            logger.debug(list(self._dataDirectoryQueue.queue))


class DataDirectoryWatcher(Observable, Observer):

    def __init__(self, settings: AutomationSettings) -> None:
        super().__init__()
        self._settings = settings
        self._dataDirectoryQueue: queue.Queue[Path] = queue.Queue()
        self._observer = watchdog.observers.Observer()

    @classmethod
    def createInstance(cls, settings: AutomationSettings) -> DataDirectoryWatcher:
        watcher = cls(settings)
        settings.watchdogDirectory.addObserver(watcher)
        watcher._updateWatch()
        return watcher

    @property
    def isAlive(self) -> bool:
        return self._observer.is_alive()

    def start(self) -> None:
        if self.isAlive:
            logger.error('Watchdog thread already started!')
        else:
            self._observer = watchdog.observers.Observer()
            self._observer.start()
            logger.debug('Watchdog thread started.')

    def stop(self) -> None:
        if self.isAlive:
            self._observer.stop()
            self._observer.join()
            logger.debug('Watchdog thread stopped.')

    def _updateWatch(self) -> None:
        if self.isAlive:
            self._observer.unschedule_all()

        observedWatch = self._observer.schedule(
            event_handler=DataDirectoryEventHandler(self._dataDirectoryQueue),
            path=self._settings.watchdogDirectory.value,
        )
        logger.debug(observedWatch)

    def update(self, observable: Observable) -> None:
        if observable is self._settings.watchdogDirectory:
            self._updateWatch()

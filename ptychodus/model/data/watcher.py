from __future__ import annotations
from pathlib import Path
import logging
import queue
import threading
import watchdog.events
import watchdog.observers

from ...api.observer import Observable, Observer
from .dataset import ActiveDiffractionDataset
from .settings import DiffractionDatasetSettings

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


class DataDirectoryWatcher(Observer):

    def __init__(self, settings: DiffractionDatasetSettings,
                 dataset: ActiveDiffractionDataset) -> None:
        super().__init__()
        self._settings = settings
        self._dataset = dataset
        self._dataDirectoryQueue: queue.Queue[Path] = queue.Queue()
        self._observer = watchdog.observers.Observer()

    @classmethod
    def createInstance(cls, settings: DiffractionDatasetSettings,
                       dataset: ActiveDiffractionDataset) -> DataDirectoryWatcher:
        watcher = cls(settings, dataset)
        settings.watchdogEnabled.addObserver(watcher)
        settings.watchdogDirectory.addObserver(watcher)
        watcher._updateWatchdogThread()
        watcher._updateWatch()
        return watcher

    def start(self) -> None:
        if self._observer.is_alive():
            logger.error('Watchdog thread already started!')
        else:
            self._observer = watchdog.observers.Observer()
            self._observer.start()
            logger.debug('Watchdog thread started.')

    def stop(self) -> None:
        if self._observer.is_alive():
            self._observer.stop()
            self._observer.join()
            logger.debug('Watchdog thread stopped.')

    def _updateWatchdogThread(self) -> None:
        if self._settings.watchdogEnabled.value:
            self.start()
        else:
            self.stop()

    def _updateWatch(self) -> None:
        if self._observer.is_alive():
            self._observer.unschedule_all()

        observedWatch = self._observer.schedule(
            event_handler=DataDirectoryEventHandler(self._dataDirectoryQueue),
            path=self._settings.watchdogDirectory.value,
        )
        logger.debug(observedWatch)

    def update(self, observable: Observable) -> None:
        if observable is self._settings.watchdogEnabled:
            self._updateWatchdogThread()
        elif observable is self._settings.watchdogDirectory:
            self._updateWatch()

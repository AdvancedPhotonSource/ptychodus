from __future__ import annotations
import logging
import threading
import watchdog.events
import watchdog.observers

from ..api.observer import Observable, Observer
from .data import DataSettings

logger = logging.getLogger(__name__)


class DataFileEventHandler(watchdog.events.PatternMatchingEventHandler):

    def __init__(self, patterns: list[str]) -> None:
        super().__init__(patterns=patterns, ignore_directories=True, case_sensitive=False)

    def on_any_event(self, event) -> None:
        logger.debug(f'{event.event_type}: {event.src_path}')


class DataDirectoryWatcher(Observer):

    def __init__(self, settings: DataSettings) -> None:
        super().__init__()
        self._settings = settings
        self._observer = watchdog.observers.Observer()

    @classmethod
    def createInstance(cls, settings: DataSettings) -> DataDirectoryWatcher:
        watcher = cls(settings)
        settings.filePath.addObserver(watcher)
        return watcher

    def start(self) -> None:
        filePath = self._settings.filePath.value

        if self._observer.is_alive():
            logger.debug('Watchdog thread is already alive!')
            return

        if filePath.is_file():
            patterns = [f'*{filePath.suffix}']
            eventHandler = DataFileEventHandler(patterns)
            directory = filePath.parent

            self._observer = watchdog.observers.Observer()
            self._observer.schedule(eventHandler, directory, recursive=False)
            self._observer.start()
            logger.debug(f'Watchdog thread is watching \"{directory}\" for {patterns}.')

    def stop(self) -> None:
        if self._observer.is_alive():
            self._observer.stop()
            self._observer.join()
            logger.debug('Watchdog thread stopped.')

    def update(self, observable: Observable) -> None:
        if observable is self._settings.filePath:
            self.stop()
            self.start()

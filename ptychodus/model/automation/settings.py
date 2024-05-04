from __future__ import annotations
from pathlib import Path

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class AutomationSettings(Observable, Observer):

    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.createGroup('Automation')
        self._settingsGroup.addObserver(self)

        self.strategy = self._settingsGroup.createStringEntry('Strategy', 'APS2ID')
        self.dataDirectory = self._settingsGroup.createPathEntry('DataDirectory',
                                                                 Path('/path/to/data'))
        self.processingIntervalInSeconds = self._settingsGroup.createIntegerEntry(
            'ProcessingIntervalInSeconds', 0)
        self.useWatchdogPollingObserver = self._settingsGroup.createBooleanEntry(
            'UseWatchdogPollingObserver', False)
        self.watchdogDelayInSeconds = self._settingsGroup.createIntegerEntry(
            'WatchdogDelayInSeconds', 15)

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()

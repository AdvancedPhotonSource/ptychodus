from __future__ import annotations
from pathlib import Path

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class AutomationSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.create_group('Automation')
        self._settingsGroup.add_observer(self)

        self.strategy = self._settingsGroup.create_string_parameter('Strategy', 'APS2ID')
        self.dataDirectory = self._settingsGroup.create_path_parameter(
            'DataDirectory', Path('/path/to/data')
        )
        self.processingIntervalInSeconds = self._settingsGroup.create_integer_parameter(
            'ProcessingIntervalInSeconds', 0
        )
        self.useWatchdogPollingObserver = self._settingsGroup.create_boolean_parameter(
            'UseWatchdogPollingObserver', False
        )
        self.watchdogDelayInSeconds = self._settingsGroup.create_integer_parameter(
            'WatchdogDelayInSeconds', 15
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notify_observers()

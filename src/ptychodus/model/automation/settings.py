from __future__ import annotations
from pathlib import Path

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class AutomationSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._group = registry.create_group('Automation')
        self._group.add_observer(self)

        self.strategy = self._group.create_string_parameter('Strategy', 'Autoload_Product')
        self.data_directory = self._group.create_path_parameter(
            'DataDirectory', Path('/path/to/data')
        )
        self.processing_interval_s = self._group.create_integer_parameter(
            'ProcessingIntervalInSeconds', 0
        )
        self.use_watchdog_polling_observer = self._group.create_boolean_parameter(
            'UseWatchdogPollingObserver', False
        )
        self.watchdog_delay_s = self._group.create_integer_parameter('WatchdogDelayInSeconds', 15)

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()

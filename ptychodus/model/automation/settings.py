from __future__ import annotations
from pathlib import Path

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry
from ptychodus.api.parametric import (
    BooleanParameter,
    IntegerParameter,
    PathParameter,
    StringParameter,
)


class AutomationSettings(Observable, Observer):

    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.createGroup("Automation")
        self._settingsGroup.addObserver(self)

        self.strategy = StringParameter(self._settingsGroup, "Strategy", "APS2ID")
        self.dataDirectory = PathParameter(self._settingsGroup, "DataDirectory",
                                           Path("/path/to/data"))
        self.processingIntervalInSeconds = IntegerParameter(self._settingsGroup,
                                                            "ProcessingIntervalInSeconds", 0)
        self.useWatchdogPollingObserver = BooleanParameter(self._settingsGroup,
                                                           "UseWatchdogPollingObserver", False)
        self.watchdogDelayInSeconds = IntegerParameter(self._settingsGroup,
                                                       "WatchdogDelayInSeconds", 15)

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()

from __future__ import annotations
from uuid import UUID

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class WorkflowSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.createGroup('Workflow')
        self._settingsGroup.addObserver(self)

        self.computeEndpointID = self._settingsGroup.createUUIDParameter(
            'ComputeEndpointID', UUID(int=0)
        )
        self.statusRefreshIntervalInSeconds = self._settingsGroup.createIntegerParameter(
            'StatusRefreshIntervalInSeconds', 10
        )

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()

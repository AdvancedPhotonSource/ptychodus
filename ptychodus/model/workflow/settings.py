from __future__ import annotations
from uuid import UUID

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class WorkflowSettings(Observable, Observer):

    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self.group = registry.createGroup('Workflow')
        self.group.addObserver(self)
        self.computeEndpointID = self.group.createUUIDEntry('ComputeEndpointID', UUID(int=0))
        self.statusRefreshIntervalInSeconds = self.group.createIntegerEntry(
            'StatusRefreshIntervalInSeconds', 10)

    def update(self, observable: Observable) -> None:
        if observable is self.group:
            self.notifyObservers()

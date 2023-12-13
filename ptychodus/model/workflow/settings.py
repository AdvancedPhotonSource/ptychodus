from __future__ import annotations
from uuid import UUID

from ...api.observer import Observable, Observer
from ...api.settings import SettingsRegistry, SettingsGroup


class WorkflowSettings(Observable, Observer):

    def __init__(self, group: SettingsGroup) -> None:
        super().__init__()
        self.group = group
        self.computeEndpointID = group.createUUIDEntry('ComputeEndpointID', UUID(int=0))
        self.statusRefreshIntervalInSeconds = group.createIntegerEntry(
            'StatusRefreshIntervalInSeconds', 10)

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> WorkflowSettings:
        settings = cls(settingsRegistry.createGroup('Workflow'))
        settings.group.addObserver(settings)
        return settings

    def update(self, observable: Observable) -> None:
        if observable is self.group:
            self.notifyObservers()

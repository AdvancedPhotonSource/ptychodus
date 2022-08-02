from __future__ import annotations
from pathlib import Path
from uuid import UUID

from ...api.observer import Observable, Observer
from ...api.settings import SettingsRegistry, SettingsGroup


class WorkflowSettings(Observable, Observer):

    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup
        self.dataSourceEndpointID = settingsGroup.createUUIDEntry(
            'DataSourceEndpointID', UUID('9c9cb97e-de86-11e6-9d15-22000a1e3b52'))
        self.dataSourcePath = settingsGroup.createPathEntry('DataSourcePath',
                                                            Path('/~/path/to/data'))
        self.dataDestinationEndpointID = settingsGroup.createUUIDEntry(
            'DataDestinationEndpointID', UUID(int=0))
        self.dataDestinationPath = settingsGroup.createPathEntry('DataDestinationPath',
                                                                 Path('/~/path/to/data'))
        self.computeEndpointID = settingsGroup.createUUIDEntry(
            'ComputeEndpointID', UUID('b35e121c-5ed6-4980-a32e-9aee09089c36'))
        self.flowID = settingsGroup.createUUIDEntry('FlowID',
                                                    UUID('1e6b4406-ee3d-4bc5-9198-74128e108111'))

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> WorkflowSettings:
        settings = cls(settingsRegistry.createGroup('Workflow'))
        settings._settingsGroup.addObserver(settings)
        return settings

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()

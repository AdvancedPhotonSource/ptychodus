from __future__ import annotations
from pathlib import Path
from uuid import UUID

from ...api.observer import Observable, Observer
from ...api.settings import SettingsRegistry, SettingsGroup


class WorkflowSettings(Observable, Observer):

    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup
        self.inputDataEndpointID = settingsGroup.createUUIDEntry(
            'InputDataEndpointID', UUID('9c9cb97e-de86-11e6-9d15-22000a1e3b52'))
        self.inputDataPath = settingsGroup.createPathEntry('InputDataPath',
                                                           Path('/~/path/to/data'))
        self.computeEndpointID = settingsGroup.createUUIDEntry(
            'ComputeEndpointID', UUID('08925f04-569f-11e7-bef8-22000b9a448b'))
        self.computeDataEndpointID = settingsGroup.createUUIDEntry('ComputeDataEndpointID',
                                                                   UUID(int=0))
        self.computeDataPath = settingsGroup.createPathEntry('ComputeDataPath',
                                                             Path('/~/path/to/data'))
        self.flowID = settingsGroup.createUUIDEntry('FlowID',
                                                    UUID('8465696e-6bd6-4c9b-b5f2-7d4a6b19c5c1'))
        self.outputDataEndpointID = settingsGroup.createUUIDEntry(
            'OutputDataEndpointID', UUID('9c9cb97e-de86-11e6-9d15-22000a1e3b52'))
        self.outputDataPath = settingsGroup.createPathEntry('OutputDataPath',
                                                            Path('/~/path/to/data'))
        self.statusRefreshIntervalInSeconds = settingsGroup.createIntegerEntry(
            'StatusRefreshIntervalInSeconds', 10)

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> WorkflowSettings:
        settings = cls(settingsRegistry.createGroup('Workflow'))
        settings._settingsGroup.addObserver(settings)
        return settings

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()

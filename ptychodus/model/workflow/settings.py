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
        self.inputDataPath = settingsGroup.createStringEntry('InputDataPath', '/~/path/to/data')
        self.computeEndpointID = settingsGroup.createUUIDEntry(
            'ComputeEndpointID', UUID('08925f04-569f-11e7-bef8-22000b9a448b'))
        self.computeDataEndpointID = settingsGroup.createUUIDEntry('ComputeDataEndpointID',
                                                                   UUID(int=0))
        self.computeDataPath = settingsGroup.createStringEntry('ComputeDataPath',
                                                               '/~/path/to/data')
        self.flowID = settingsGroup.createUUIDEntry('FlowID',
                                                    UUID('9fb405f9-be63-4ce7-9605-faf62752bea2'))
        self.reconstructActionID = settingsGroup.createUUIDEntry(
            'ReconstructActionID', UUID('0148eb97-eac0-4afa-be07-131d18cb5243'))
        self.outputDataEndpointID = settingsGroup.createUUIDEntry(
            'OutputDataEndpointID', UUID('9c9cb97e-de86-11e6-9d15-22000a1e3b52'))
        self.outputDataPath = settingsGroup.createStringEntry('OutputDataPath', '/~/path/to/data')
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

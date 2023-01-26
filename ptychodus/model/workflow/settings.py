from __future__ import annotations
from pathlib import Path
from uuid import UUID

from ...api.observer import Observable, Observer
from ...api.settings import SettingsRegistry, SettingsGroup


class WorkflowSettings(Observable, Observer):

    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup

        self.inputDataEndpointID = settingsGroup.createUUIDEntry('InputDataEndpointID',
                                                                 UUID(int=0))
        self.inputDataGlobusPath = settingsGroup.createStringEntry('InputDataGlobusPath',
                                                                   '/~/path/to/input/data')
        self.inputDataPosixPath = settingsGroup.createPathEntry('InputDataPosixPath',
                                                                Path.home() / 'ptychodus-staging')
        self.computeFuncXEndpointID = settingsGroup.createUUIDEntry('ComputeFuncXEndpointID',
                                                                    UUID(int=0))
        self.computeDataEndpointID = settingsGroup.createUUIDEntry('ComputeDataEndpointID',
                                                                   UUID(int=0))
        self.computeDataGlobusPath = settingsGroup.createStringEntry('ComputeDataGlobusPath',
                                                                     '/~/path/to/compute/data')
        self.computeDataPosixPath = settingsGroup.createPathEntry('ComputeDataPosixPath',
                                                                  Path('/path/to/compute/data'))
        self.outputDataEndpointID = settingsGroup.createUUIDEntry('OutputDataEndpointID',
                                                                  UUID(int=0))
        self.outputDataGlobusPath = settingsGroup.createStringEntry('OutputDataGlobusPath',
                                                                    '/~/path/to/output/data')
        self.outputDataPosixPath = settingsGroup.createPathEntry('OutputDataPosixPath',
                                                                 Path('/path/to/output/data'))
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

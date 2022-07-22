from __future__ import annotations
from pathlib import Path
from uuid import UUID
import logging

from ..api.observer import Observable, Observer
from ..api.settings import SettingsRegistry, SettingsGroup

logger = logging.getLogger(__name__)


class WorkflowSettings(Observable, Observer):

    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup
        self.dataSourceEndpointUUID = settingsGroup.createUUIDEntry('DataSourceEndpointUUID',
                                                                    UUID(int=0))
        self.dataSourcePath = settingsGroup.createPathEntry('DataSourcePath',
                                                            Path('/~/path/to/data'))
        self.dataDestinationEndpointUUID = settingsGroup.createUUIDEntry(
            'DataDestinationEndpointUUID', UUID(int=0))
        self.dataDestinationPath = settingsGroup.createPathEntry('DataDestinationPath',
                                                                 Path('/~/path/to/data'))
        self.computeEndpointUUID = settingsGroup.createUUIDEntry('ComputeEndpointUUID',
                                                                 UUID(int=0))

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> WorkflowSettings:
        settings = cls(settingsRegistry.createGroup('Workflow'))
        settings._settingsGroup.addObserver(settings)
        return settings

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()


class WorkflowPresenter(Observer, Observable):

    def __init__(self, settings: WorkflowSettings) -> None:
        super().__init__()
        self._settings = settings

    @classmethod
    def createInstance(cls, settings: WorkflowSettings) -> WorkflowPresenter:
        presenter = cls(settings)
        settings.addObserver(presenter)
        return presenter

    def setDataSourceEndpointUUID(self, endpointUUID: UUID) -> None:
        self._settings.dataSourceEndpointUUID.value = endpointUUID

    def getDataSourceEndpointUUID(self) -> UUID:
        return self._settings.dataSourceEndpointUUID.value

    def setDataSourcePath(self, dataSourcePath: Path) -> None:
        self._settings.dataSourcePath.value = dataSourcePath

    def getDataSourcePath(self) -> Path:
        return self._settings.dataSourcePath.value

    def setDataDestinationEndpointUUID(self, endpointUUID: UUID) -> None:
        self._settings.dataDestinationEndpointUUID.value = endpointUUID

    def getDataDestinationEndpointUUID(self) -> UUID:
        return self._settings.dataDestinationEndpointUUID.value

    def setDataDestinationPath(self, dataDestinationPath: Path) -> None:
        self._settings.dataDestinationPath.value = dataDestinationPath

    def getDataDestinationPath(self) -> Path:
        return self._settings.dataDestinationPath.value

    def setComputeEndpointUUID(self, endpointUUID: UUID) -> None:
        self._settings.computeEndpointUUID.value = endpointUUID

    def getComputeEndpointUUID(self) -> UUID:
        return self._settings.computeEndpointUUID.value

    def launchWorkflow(self) -> None:
        logger.info('Launch workflow!')

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notifyObservers()

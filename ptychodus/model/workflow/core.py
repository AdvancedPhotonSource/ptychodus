from __future__ import annotations
from pathlib import Path
from typing import Optional
from uuid import UUID
import logging

from ...api.observer import Observable, Observer
from ...api.settings import SettingsRegistry
from .settings import WorkflowSettings

try:
    from .globusClient import GlobusWorkflowClient as WorkflowClient
    from .globusClient import GlobusWorkflowClientBuilder as WorkflowClientBuilder
except ModuleNotFoundError:
    from .client import WorkflowClient
    from .client import WorkflowClientBuilder

logger = logging.getLogger(__name__)


class WorkflowPresenter(Observer, Observable):

    def __init__(self, settings: WorkflowSettings) -> None:
        super().__init__()
        self._settings = settings
        self._clientBuilder = WorkflowClientBuilder(settings)
        self._client: Optional[WorkflowClient] = None

    @classmethod
    def createInstance(cls, settings: WorkflowSettings) -> WorkflowPresenter:
        presenter = cls(settings)
        settings.addObserver(presenter)
        return presenter

    def isAuthorized(self) -> bool:
        return not (self._client is None)

    def setAuthorizationCode(self, authCode: str) -> None:
        self._client = self._clientBuilder.build(authCode)
        self.notifyObservers()

    def getAuthorizeURL(self) -> str:
        return self._clientBuilder.getAuthorizeURL()

    def setDataSourceEndpointID(self, endpointID: UUID) -> None:
        self._settings.dataSourceEndpointID.value = endpointID

    def getDataSourceEndpointID(self) -> UUID:
        return self._settings.dataSourceEndpointID.value

    def setDataSourcePath(self, dataSourcePath: Path) -> None:
        self._settings.dataSourcePath.value = dataSourcePath

    def getDataSourcePath(self) -> Path:
        return self._settings.dataSourcePath.value

    def setDataDestinationEndpointID(self, endpointID: UUID) -> None:
        self._settings.dataDestinationEndpointID.value = endpointID

    def getDataDestinationEndpointID(self) -> UUID:
        return self._settings.dataDestinationEndpointID.value

    def setDataDestinationPath(self, dataDestinationPath: Path) -> None:
        self._settings.dataDestinationPath.value = dataDestinationPath

    def getDataDestinationPath(self) -> Path:
        return self._settings.dataDestinationPath.value

    def setComputeEndpointID(self, endpointID: UUID) -> None:
        self._settings.computeEndpointID.value = endpointID

    def getComputeEndpointID(self) -> UUID:
        return self._settings.computeEndpointID.value

    def setFlowID(self, endpointID: UUID) -> None:
        self._settings.flowID.value = endpointID

    def getFlowID(self) -> UUID:
        return self._settings.flowID.value

    def listFlows(self) -> None:
        if self._client:
            self._client.listFlows()

    def listFlowRuns(self) -> None:
        if self._client:
            self._client.listFlowRuns()

    def deployFlow(self) -> None:
        if self._client:
            self._client.deployFlow()

    def runFlow(self) -> None:
        if self._client:
            self._client.runFlow()

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notifyObservers()


class WorkflowCore:

    def __init__(self, settingsRegistry: SettingsRegistry) -> None:
        self._settings = WorkflowSettings.createInstance(settingsRegistry)
        self.presenter = WorkflowPresenter.createInstance(self._settings)

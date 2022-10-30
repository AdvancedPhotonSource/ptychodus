from __future__ import annotations
from pathlib import Path
from typing import Optional
from uuid import UUID
import logging

from ...api.observer import Observable, Observer
from ...api.settings import SettingsRegistry
from .settings import WorkflowSettings
from .client import WorkflowClient, WorkflowClientBuilder, WorkflowRun

logger = logging.getLogger(__name__)


class WorkflowPresenter(Observable, Observer):

    def __init__(self, settings: WorkflowSettings) -> None:
        super().__init__()
        self._settings = settings

        try:
            from .globusClient import GlobusWorkflowClientBuilder
            self._clientBuilder: WorkflowClientBuilder = GlobusWorkflowClientBuilder(settings)
        except ModuleNotFoundError:
            self._clientBuilder = WorkflowClientBuilder()

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

    def setInputDataEndpointID(self, endpointID: UUID) -> None:
        self._settings.inputDataEndpointID.value = endpointID

    def getInputDataEndpointID(self) -> UUID:
        return self._settings.inputDataEndpointID.value

    def setInputDataPath(self, inputDataPath: str) -> None:
        self._settings.inputDataPath.value = inputDataPath

    def getInputDataPath(self) -> str:
        return self._settings.inputDataPath.value

    def setOutputDataEndpointID(self, endpointID: UUID) -> None:
        self._settings.outputDataEndpointID.value = endpointID

    def getOutputDataEndpointID(self) -> UUID:
        return self._settings.outputDataEndpointID.value

    def setOutputDataPath(self, outputDataPath: str) -> None:
        self._settings.outputDataPath.value = outputDataPath

    def getOutputDataPath(self) -> str:
        return self._settings.outputDataPath.value

    def setComputeEndpointID(self, endpointID: UUID) -> None:
        self._settings.computeEndpointID.value = endpointID

    def getComputeEndpointID(self) -> UUID:
        return self._settings.computeEndpointID.value

    def setFlowID(self, flowID: UUID) -> None:
        self._settings.flowID.value = flowID

    def getFlowID(self) -> UUID:
        return self._settings.flowID.value

    def setReconstructActionID(self, actionID: UUID) -> None:
        self._settings.reconstructActionID.value = actionID

    def getReconstructActionID(self) -> UUID:
        return self._settings.reconstructActionID.value

    def setComputeDataEndpointID(self, endpointID: UUID) -> None:
        self._settings.computeDataEndpointID.value = endpointID

    def getComputeDataEndpointID(self) -> UUID:
        return self._settings.computeDataEndpointID.value

    def setComputeDataPath(self, computeDataPath: str) -> None:
        self._settings.computeDataPath.value = computeDataPath

    def getComputeDataPath(self) -> str:
        return self._settings.computeDataPath.value

    def setStatusRefreshIntervalInSeconds(self, seconds: int) -> None:
        self._settings.statusRefreshIntervalInSeconds.value = seconds

    def getStatusRefreshIntervalInSeconds(self) -> int:
        return self._settings.statusRefreshIntervalInSeconds.value

    def listFlowRuns(self) -> list[WorkflowRun]:
        flowRuns: list[WorkflowRun] = list()

        if self._client:
            flowRuns.extend(self._client.listFlowRuns())

        return flowRuns

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

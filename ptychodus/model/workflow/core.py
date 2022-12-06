from __future__ import annotations
from pathlib import Path
from typing import Optional
from uuid import UUID
import logging

from ...api.observer import Observable, Observer
from ...api.settings import SettingsRegistry
from .api import WorkflowClient, WorkflowRun
from .settings import WorkflowSettings

logger = logging.getLogger(__name__)

# TODO ideally initial_authorizers will contain all of the scopes needed so no login is needed
#from globus_automate_client.flows_client import (MANAGE_FLOWS_SCOPE, RUN_FLOWS_SCOPE,
#                                                 RUN_STATUS_SCOPE, VIEW_FLOWS_SCOPE)

#FLOW_ID = str(self._settings.flowID.value)
#FLOW_ID_ = FLOW_ID.replace('-', '_')

#requestedScopes = [
#    # Automate scopes
#    MANAGE_FLOWS_SCOPE,
#    RUN_FLOWS_SCOPE,
#    RUN_STATUS_SCOPE,
#    VIEW_FLOWS_SCOPE,
#
#    # Flow scope
#    f'https://auth.globus.org/scopes/{FLOW_ID}/flow_{FLOW_ID_}_user',
#]

# def _reauthorizeCallback(self, scopes: list[str]) -> ScopeAuthorizerMapping:
#     # TODO complete a Globus Auth flow
#     # TODO secure tokens for future invocations
#     self.reauthorize(scopes)
#     # FIXME controller uses authService to present GUI for user to reauth
#     # FIXME wait for authorizers
#     return authorizers

# def buildLoginManager(self, authCode: str) -> BaseLoginManager:
#     return CallbackLoginManager(
#         initial_authorizers=authorizers,
#         callback=self._reauthorizeCallback,
#     )


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
            self._client.runFlow(label='Ptychodus')  # TODO label

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notifyObservers()


class WorkflowCore:

    def __init__(self, settingsRegistry: SettingsRegistry) -> None:
        self._settings = WorkflowSettings.createInstance(settingsRegistry)
        self.presenter = WorkflowPresenter.createInstance(self._settings)

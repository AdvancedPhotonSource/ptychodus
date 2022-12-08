from __future__ import annotations
from pathlib import Path
from typing import Optional
from uuid import UUID
import logging

from ...api.observer import Observable, Observer
from ...api.settings import SettingsRegistry
from .api import WorkflowAuthorizerRepository, WorkflowClient, WorkflowRun
from .settings import WorkflowSettings

logger = logging.getLogger(__name__)


class WorkflowPresenter(Observable, Observer):

    def __init__(self, settings: WorkflowSettings,
                 authorizerRepository: WorkflowAuthorizerRepository,
                 client: WorkflowClient) -> None:
        super().__init__()
        self._settings = settings
        self._authorizerRepository = authorizerRepository
        self._client = client

    @classmethod
    def createInstance(cls, settings: WorkflowSettings,
                       authorizerRepository: WorkflowAuthorizerRepository,
                       client: WorkflowClient) -> WorkflowPresenter:
        presenter = cls(settings, authorizerRepository, client)
        settings.addObserver(presenter)
        return presenter

    def isAuthorized(self) -> bool:
        return self._authorizerRepository.isAuthorized

    def getAuthorizeURL(self) -> str:
        return self._authorizerRepository.getAuthorizeURL()

    def setCodeFromAuthorizeURL(self, code: str) -> None:
        self._authorizerRepository.setCodeFromAuthorizeURL(code)

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
        return self._client.listFlowRuns()

    def runFlow(self) -> None:
        self._client.runFlow(label='Ptychodus')  # TODO label

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notifyObservers()


class WorkflowCore:

    def __init__(self, settingsRegistry: SettingsRegistry) -> None:
        self._settings = WorkflowSettings.createInstance(settingsRegistry)

        try:
            from .authorizerRepository import GlobusAuthorizerRepository
            from .client import GlobusClient
        except ModuleNotFoundError:
            logger.info('Globus not found.')

            from .null import NullAuthorizerRepository, NullClient
            self._authorizerRepository: WorkflowAuthorizerRepository = NullAuthorizerRepository()
            self._client: WorkflowClient = NullClient()
        else:
            globusAuthorizerRepository = GlobusAuthorizerRepository()
            self._authorizerRepository = globusAuthorizerRepository
            self._client = GlobusClient(self._settings, globusAuthorizerRepository)

        self.presenter = WorkflowPresenter.createInstance(self._settings,
                                                          self._authorizerRepository, self._client)

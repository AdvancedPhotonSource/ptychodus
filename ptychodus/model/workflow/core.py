from __future__ import annotations
from pathlib import Path
from typing import Optional, TypeAlias
from uuid import UUID
import logging

from ...api.observer import Observable, Observer
from ...api.settings import SettingsRegistry
from ..statefulCore import StateDataRegistry
from .api import WorkflowAuthorizerRepository, WorkflowClient, WorkflowRun
from .settings import WorkflowSettings

logger = logging.getLogger(__name__)

WorkflowAuthorizationPresenter: TypeAlias = WorkflowAuthorizerRepository
WorkflowExecutionPresenter: TypeAlias = WorkflowClient


class WorkflowParametersPresenter(Observable, Observer):

    def __init__(self, settings: WorkflowSettings) -> None:
        super().__init__()
        self._settings = settings

    @classmethod
    def createInstance(cls, settings: WorkflowSettings) -> WorkflowParametersPresenter:
        presenter = cls(settings)
        settings.addObserver(presenter)
        return presenter

    def setInputDataEndpointID(self, endpointID: UUID) -> None:
        self._settings.inputDataEndpointID.value = endpointID

    def getInputDataEndpointID(self) -> UUID:
        return self._settings.inputDataEndpointID.value

    def setInputDataGlobusPath(self, inputDataGlobusPath: str) -> None:
        self._settings.inputDataGlobusPath.value = inputDataGlobusPath

    def getInputDataGlobusPath(self) -> str:
        return self._settings.inputDataGlobusPath.value

    def setInputDataPosixPath(self, inputDataPosixPath: str) -> None:
        self._settings.inputDataPosixPath.value = inputDataPosixPath

    def getInputDataPosixPath(self) -> str:
        return self._settings.inputDataPosixPath.value

    def setComputeFuncXEndpointID(self, endpointID: UUID) -> None:
        self._settings.computeFuncXEndpointID.value = endpointID

    def getComputeFuncXEndpointID(self) -> UUID:
        return self._settings.computeFuncXEndpointID.value

    def setComputeDataEndpointID(self, endpointID: UUID) -> None:
        self._settings.computeDataEndpointID.value = endpointID

    def getComputeDataEndpointID(self) -> UUID:
        return self._settings.computeDataEndpointID.value

    def setComputeDataGlobusPath(self, computeDataGlobusPath: str) -> None:
        self._settings.computeDataGlobusPath.value = computeDataGlobusPath

    def getComputeDataGlobusPath(self) -> str:
        return self._settings.computeDataGlobusPath.value

    def setComputeDataPosixPath(self, computeDataPosixPath: str) -> None:
        self._settings.computeDataPosixPath.value = computeDataPosixPath

    def getComputeDataPosixPath(self) -> str:
        return self._settings.computeDataPosixPath.value

    def setOutputDataEndpointID(self, endpointID: UUID) -> None:
        self._settings.outputDataEndpointID.value = endpointID

    def getOutputDataEndpointID(self) -> UUID:
        return self._settings.outputDataEndpointID.value

    def setOutputDataGlobusPath(self, outputDataGlobusPath: str) -> None:
        self._settings.outputDataGlobusPath.value = outputDataGlobusPath

    def getOutputDataGlobusPath(self) -> str:
        return self._settings.outputDataGlobusPath.value

    def setOutputDataPosixPath(self, outputDataPosixPath: str) -> None:
        self._settings.outputDataPosixPath.value = outputDataPosixPath

    def getOutputDataPosixPath(self) -> str:
        return self._settings.outputDataPosixPath.value

    def setStatusRefreshIntervalInSeconds(self, seconds: int) -> None:
        self._settings.statusRefreshIntervalInSeconds.value = seconds

    def getStatusRefreshIntervalInSeconds(self) -> int:
        return self._settings.statusRefreshIntervalInSeconds.value

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notifyObservers()


class WorkflowCore:

    def __init__(self, settingsRegistry: SettingsRegistry,
                 stateDataRegistry: StateDataRegistry) -> None:
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
            self._client = GlobusClient(self._settings, settingsRegistry, stateDataRegistry,
                                        globusAuthorizerRepository)

        self.parametersPresenter = WorkflowParametersPresenter.createInstance(self._settings)

    @property
    def authorizationPresenter(self) -> WorkflowAuthorizationPresenter:
        return self._authorizerRepository

    @property
    def executionPresenter(self) -> WorkflowExecutionPresenter:
        return self._client

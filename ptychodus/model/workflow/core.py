from __future__ import annotations
from collections.abc import Sequence
from pathlib import Path
from typing import Optional
from uuid import UUID
import logging

from ...api.observer import Observable, Observer
from ...api.settings import SettingsRegistry
from ..statefulCore import StateDataRegistry
from .api import WorkflowAuthorizer, WorkflowExecutor, WorkflowRun
from .settings import WorkflowSettings

logger = logging.getLogger(__name__)


class WorkflowAuthorizationPresenter(WorkflowAuthorizer):

    def __init__(self, authorizer: Optional[WorkflowAuthorizer]) -> None:
        self._authorizer = authorizer

    @property
    def isAuthorized(self) -> bool:
        if self._authorizer:
            return self._authorizer.isAuthorized
        else:
            return True

    def getAuthorizeURL(self) -> str:
        if self._authorizer:
            return self._authorizer.getAuthorizeURL()
        else:
            return 'https://aps.anl.gov'

    def setCodeFromAuthorizeURL(self, code: str) -> None:
        if self._authorizer:
            self._authorizer.setCodeFromAuthorizeURL(code)
        else:
            logger.error('Cannot set auth code with null authorizer!')


class WorkflowExecutionPresenter(WorkflowExecutor):

    def __init__(self, client: Optional[WorkflowExecutor]) -> None:
        self._client = client

    def listFlowRuns(self) -> Sequence[WorkflowRun]:
        if self._client:
            return self._client.listFlowRuns()
        else:
            flowRuns: list[WorkflowRun] = list()
            return flowRuns

    def runFlow(self, label: str) -> None:
        if self._client:
            self._client.runFlow(label)
        else:
            logger.error('Cannot run flow with null executor!')

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass


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

    def setInputDataPosixPath(self, inputDataPosixPath: Path) -> None:
        self._settings.inputDataPosixPath.value = inputDataPosixPath

    def getInputDataPosixPath(self) -> Path:
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
        self._authorizer: Optional[WorkflowAuthorizer] = None
        self._executor: Optional[WorkflowExecutor] = None

        try:
            from .globus import GlobusWorkflowAuthorizer, GlobusWorkflowExecutor
        except ModuleNotFoundError:
            logger.info('Globus not found.')
        else:
            globusAuthorizer = GlobusWorkflowAuthorizer()
            self._authorizer = globusAuthorizer
            self._executor = GlobusWorkflowExecutor(self._settings, globusAuthorizer,
                                                    settingsRegistry, stateDataRegistry)

        self.authorizationPresenter = WorkflowAuthorizationPresenter(self._authorizer)
        self.executionPresenter = WorkflowExecutionPresenter(self._executor)
        self.parametersPresenter = WorkflowParametersPresenter.createInstance(self._settings)

    def start(self) -> None:
        if self._executor:
            self._executor.start()

    def stop(self) -> None:
        if self._executor:
            self._executor.stop()

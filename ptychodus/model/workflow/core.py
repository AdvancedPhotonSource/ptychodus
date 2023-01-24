from __future__ import annotations
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, overload
from uuid import UUID
import logging
import threading

from ...api.geometry import Interval
from ...api.observer import Observable, Observer
from ...api.settings import SettingsRegistry
from ..statefulCore import StateDataRegistry
from .authorizer import WorkflowAuthorizer
from .executor import WorkflowExecutor
from .settings import WorkflowSettings
from .status import WorkflowStatus, WorkflowStatusRepository

logger = logging.getLogger(__name__)


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

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notifyObservers()


class WorkflowAuthorizationPresenter:

    def __init__(self, authorizer: WorkflowAuthorizer) -> None:
        self._authorizer = authorizer

    @property
    def isAuthorized(self) -> bool:
        return self._authorizer.isAuthorized

    def getAuthorizeURL(self) -> str:
        return self._authorizer.getAuthorizeURL()

    def setCodeFromAuthorizeURL(self, code: str) -> None:
        self._authorizer.setCodeFromAuthorizeURL(code)


class WorkflowStatusPresenter:

    def __init__(self, settings: WorkflowSettings,
                 statusRepository: WorkflowStatusRepository) -> None:
        self._settings = settings
        self._statusRepository = statusRepository

    def getRefreshIntervalLimitsInSeconds(self) -> Interval[int]:
        return Interval[int](10, 86400)

    def getRefreshIntervalInSeconds(self) -> int:
        limits = self.getRefreshIntervalLimitsInSeconds()
        return limits.clamp(self._settings.statusRefreshIntervalInSeconds.value)

    def setRefreshIntervalInSeconds(self, seconds: int) -> None:
        self._settings.statusRefreshIntervalInSeconds.value = seconds

    @overload
    def __getitem__(self, index: int) -> WorkflowStatus:
        ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[WorkflowStatus]:
        ...

    def __getitem__(self, index: Union[int, slice]) -> \
            Union[WorkflowStatus, Sequence[WorkflowStatus]]:
        return self._statusRepository[index]

    def __len__(self) -> int:
        return len(self._statusRepository)

    def getStatusDateTime(self) -> datetime:
        return self._statusRepository.getStatusDateTime()

    def refreshStatus(self) -> None:
        self._statusRepository.refreshStatus()


class WorkflowExecutionPresenter:

    def __init__(self, executor: WorkflowExecutor) -> None:
        self._executor = executor

    def runFlow(self, flowLabel: str) -> None:
        self._executor.runFlow(flowLabel=flowLabel)


class WorkflowCore:

    def __init__(self, settingsRegistry: SettingsRegistry,
                 stateDataRegistry: StateDataRegistry) -> None:
        self._settings = WorkflowSettings.createInstance(settingsRegistry)
        self._authorizer = WorkflowAuthorizer()
        self._statusRepository = WorkflowStatusRepository()
        self._executor = WorkflowExecutor(self._settings, settingsRegistry, stateDataRegistry)
        self._thread: Optional[threading.Thread] = None

        try:
            from .globus import GlobusWorkflowThread
        except ModuleNotFoundError:
            logger.info('Globus not found.')
            # FIXME hide workflow tab and don't break if globus not installed
        else:
            self._thread = GlobusWorkflowThread(self._authorizer, self._statusRepository,
                                                self._executor)

        self.parametersPresenter = WorkflowParametersPresenter.createInstance(self._settings)
        self.authorizationPresenter = WorkflowAuthorizationPresenter(self._authorizer)
        self.statusPresenter = WorkflowStatusPresenter(self._settings, self._statusRepository)
        self.executionPresenter = WorkflowExecutionPresenter(self._executor)

    def start(self) -> None:
        logger.info('Starting workflow thread...')

        if self._thread:
            self._thread.start()

        logger.info('Workflow thread started.')

    def stop(self) -> None:
        logger.info('Stopping workflow thread...')
        self._executor.jobQueue.join()
        self._authorizer.shutdownEvent.set()

        if self._thread:
            self._thread.join()

        logger.info('Workflow thread stopped.')

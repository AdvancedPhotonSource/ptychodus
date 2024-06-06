from __future__ import annotations
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import overload
from uuid import UUID
import logging
import threading

from ptychodus.api.geometry import Interval
from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry

from ..patterns import PatternsAPI
from ..product import ObjectAPI, ProbeAPI, ProductAPI, ScanAPI
from .api import ConcreteWorkflowAPI
from .authorizer import WorkflowAuthorizer
from .executor import WorkflowExecutor
from .locator import DataLocator, OutputDataLocator, SimpleDataLocator
from .settings import WorkflowSettings
from .status import WorkflowStatus, WorkflowStatusRepository

logger = logging.getLogger(__name__)


class WorkflowParametersPresenter(Observable, Observer):

    def __init__(self, settings: WorkflowSettings, inputDataLocator: DataLocator,
                 computeDataLocator: DataLocator, outputDataLocator: OutputDataLocator) -> None:
        super().__init__()
        self._settings = settings
        self._inputDataLocator = inputDataLocator
        self._computeDataLocator = computeDataLocator
        self._outputDataLocator = outputDataLocator

    @classmethod
    def createInstance(cls, settings: WorkflowSettings, inputDataLocator: DataLocator,
                       computeDataLocator: DataLocator,
                       outputDataLocator: OutputDataLocator) -> WorkflowParametersPresenter:
        presenter = cls(settings, inputDataLocator, computeDataLocator, outputDataLocator)
        settings.addObserver(presenter)
        inputDataLocator.addObserver(presenter)
        computeDataLocator.addObserver(presenter)
        outputDataLocator.addObserver(presenter)
        return presenter

    def setInputDataEndpointID(self, endpointID: UUID) -> None:
        self._inputDataLocator.setEndpointID(endpointID)

    def getInputDataEndpointID(self) -> UUID:
        return self._inputDataLocator.getEndpointID()

    def setInputDataGlobusPath(self, globusPath: str) -> None:
        self._inputDataLocator.setGlobusPath(globusPath)

    def getInputDataGlobusPath(self) -> str:
        return self._inputDataLocator.getGlobusPath()

    def setInputDataPosixPath(self, posixPath: Path) -> None:
        self._inputDataLocator.setPosixPath(posixPath)

    def getInputDataPosixPath(self) -> Path:
        return self._inputDataLocator.getPosixPath()

    def setComputeEndpointID(self, endpointID: UUID) -> None:
        self._settings.computeEndpointID.value = endpointID

    def getComputeEndpointID(self) -> UUID:
        return self._settings.computeEndpointID.value

    def setComputeDataEndpointID(self, endpointID: UUID) -> None:
        self._computeDataLocator.setEndpointID(endpointID)

    def getComputeDataEndpointID(self) -> UUID:
        return self._computeDataLocator.getEndpointID()

    def setComputeDataGlobusPath(self, globusPath: str) -> None:
        self._computeDataLocator.setGlobusPath(globusPath)

    def getComputeDataGlobusPath(self) -> str:
        return self._computeDataLocator.getGlobusPath()

    def setComputeDataPosixPath(self, posixPath: Path) -> None:
        self._computeDataLocator.setPosixPath(posixPath)

    def getComputeDataPosixPath(self) -> Path:
        return self._computeDataLocator.getPosixPath()

    def setRoundTripEnabled(self, enable: bool) -> None:
        self._outputDataLocator.setRoundTripEnabled(enable)

    def isRoundTripEnabled(self) -> bool:
        return self._outputDataLocator.isRoundTripEnabled()

    def setOutputDataEndpointID(self, endpointID: UUID) -> None:
        self._outputDataLocator.setEndpointID(endpointID)

    def getOutputDataEndpointID(self) -> UUID:
        return self._outputDataLocator.getEndpointID()

    def setOutputDataGlobusPath(self, globusPath: str) -> None:
        self._outputDataLocator.setGlobusPath(globusPath)

    def getOutputDataGlobusPath(self) -> str:
        return self._outputDataLocator.getGlobusPath()

    def setOutputDataPosixPath(self, posixPath: Path) -> None:
        self._outputDataLocator.setPosixPath(posixPath)

    def getOutputDataPosixPath(self) -> Path:
        return self._outputDataLocator.getPosixPath()

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notifyObservers()
        elif observable is self._inputDataLocator:
            self.notifyObservers()
        elif observable is self._computeDataLocator:
            self.notifyObservers()
        elif observable is self._outputDataLocator:
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

    def __getitem__(self, index: int | slice) -> WorkflowStatus | Sequence[WorkflowStatus]:
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

    def runFlow(self, inputProductIndex: int) -> None:
        self._executor.runFlow(inputProductIndex)


class WorkflowCore:

    def __init__(self, settingsRegistry: SettingsRegistry, patternsAPI: PatternsAPI,
                 productAPI: ProductAPI, scanAPI: ScanAPI, probeAPI: ProbeAPI,
                 objectAPI: ObjectAPI) -> None:
        self._settings = WorkflowSettings(settingsRegistry)
        self._inputDataLocator = SimpleDataLocator.createInstance(self._settings.group, 'Input')
        self._computeDataLocator = SimpleDataLocator.createInstance(self._settings.group,
                                                                    'Compute')
        self._outputDataLocator = OutputDataLocator.createInstance(self._settings.group, 'Output',
                                                                   self._inputDataLocator)
        self._authorizer = WorkflowAuthorizer()
        self._statusRepository = WorkflowStatusRepository()
        self._executor = WorkflowExecutor(self._settings, self._inputDataLocator,
                                          self._computeDataLocator, self._outputDataLocator,
                                          settingsRegistry, patternsAPI, productAPI)
        self.workflowAPI = ConcreteWorkflowAPI(settingsRegistry, patternsAPI, productAPI, scanAPI,
                                               probeAPI, objectAPI, self._executor)
        self._thread: threading.Thread | None = None

        try:
            from .globus import GlobusWorkflowThread
        except ModuleNotFoundError:
            logger.info('Globus not found.')
        else:
            self._thread = GlobusWorkflowThread.createInstance(self._authorizer,
                                                               self._statusRepository,
                                                               self._executor)

        self.parametersPresenter = WorkflowParametersPresenter.createInstance(
            self._settings, self._inputDataLocator, self._computeDataLocator,
            self._outputDataLocator)
        self.authorizationPresenter = WorkflowAuthorizationPresenter(self._authorizer)
        self.statusPresenter = WorkflowStatusPresenter(self._settings, self._statusRepository)
        self.executionPresenter = WorkflowExecutionPresenter(self._executor)

    @property
    def areWorkflowsSupported(self) -> bool:
        return (self._thread is not None)

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

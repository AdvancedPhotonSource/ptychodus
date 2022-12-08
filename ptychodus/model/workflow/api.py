from abc import ABC, abstractmethod, abstractproperty
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


class WorkflowAuthorizerRepository(ABC):

    @abstractproperty
    def isAuthorized(self) -> bool:
        pass

    @abstractmethod
    def getAuthorizeURL(self) -> str:
        pass

    @abstractmethod
    def setCodeFromAuthorizeURL(self, code: str) -> None:
        pass


@dataclass(frozen=True)
class WorkflowRun:
    label: str
    startTime: str
    completionTime: str
    status: str
    action: str
    runID: str
    runURL: str


class WorkflowClient(ABC):

    @abstractmethod
    def listFlowRuns(self) -> list[WorkflowRun]:
        pass

    @abstractmethod
    def runFlow(self, label: str) -> None:
        pass

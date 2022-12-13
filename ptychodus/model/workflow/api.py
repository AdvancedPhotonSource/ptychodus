from abc import ABC, abstractmethod, abstractproperty
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


class WorkflowAuthorizer(ABC):

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


class WorkflowExecutor(ABC):

    @abstractmethod
    def listFlowRuns(self) -> Sequence[WorkflowRun]:
        pass

    @abstractmethod
    def runFlow(self, label: str) -> None:
        pass


class WorkflowThread(ABC):

    @abstractmethod
    def listFlowRuns(self) -> Sequence[WorkflowRun]:
        pass

    @abstractmethod
    def runFlow(self, label: str, flowInput: Mapping[str, Any]) -> None:
        pass

    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass

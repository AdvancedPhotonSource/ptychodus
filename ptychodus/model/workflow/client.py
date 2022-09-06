from dataclasses import dataclass
from uuid import UUID

from .settings import WorkflowSettings


@dataclass(frozen=True)
class WorkflowRun:
    # TODO typing
    label: str
    startTime: str
    completionTime: str
    status: str
    action: str
    runID: str
    runURL: str


class WorkflowClient:

    def listFlowRuns(self) -> list[WorkflowRun]:
        flowRuns: list[WorkflowRun] = list()
        return flowRuns

    def runFlow(self) -> None:
        pass


class WorkflowClientBuilder:

    def __init__(self, settings: WorkflowSettings) -> None:
        self._settings = settings

    def getAuthorizeURL(self) -> str:
        return str()

    def build(self, authCode: str) -> WorkflowClient:
        return WorkflowClient()
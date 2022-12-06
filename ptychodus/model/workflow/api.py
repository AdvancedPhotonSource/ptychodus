from dataclasses import dataclass


@dataclass(frozen=True)
class WorkflowRun:
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

    def runFlow(self, label: str) -> None:
        pass

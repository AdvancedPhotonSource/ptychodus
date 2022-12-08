from .api import WorkflowAuthorizerRepository, WorkflowClient, WorkflowRun


class NullAuthorizerRepository(WorkflowAuthorizerRepository):

    @property
    def isAuthorized(self) -> bool:
        return True

    def getAuthorizeURL(self) -> str:
        return str()

    def setCodeFromAuthorizeURL(self, code: str) -> None:
        pass


class NullClient(WorkflowClient):

    def listFlowRuns(self) -> list[WorkflowRun]:
        flowRuns: list[WorkflowRun] = list()
        return flowRuns

    def runFlow(self, label: str) -> None:
        pass

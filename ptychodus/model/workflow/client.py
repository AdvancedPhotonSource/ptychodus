from .settings import WorkflowSettings


class WorkflowClient:

    def listFlows(self) -> None:
        pass

    def listFlowRuns(self) -> None:
        pass

    def deployFlow(self) -> None:
        pass

    def runFlow(self) -> None:
        pass


class WorkflowClientBuilder:

    def __init__(self, settings: WorkflowSettings) -> None:
        self._settings = settings

    def getAuthorizeURL(self) -> str:
        return str()

    def build(self, authCode: str) -> WorkflowClient:
        return WorkflowClient()

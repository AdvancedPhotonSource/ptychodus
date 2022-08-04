from .settings import WorkflowSettings


class WorkflowClient:

    def printFlows(self) -> None:
        pass

    def deployFlow(self) -> None:
        pass

    def runFlow(self) -> None:
        pass


class WorkflowClientBuilder:

    def __init__(self, settings: WorkflowSettings) -> None:
        pass

    def getAuthorizeURL(self) -> str:
        return str()

    def build(self, authCode: str) -> WorkflowClient:
        return WorkflowClient()

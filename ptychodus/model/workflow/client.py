try:
    from ._globusClient import WorkflowClient, WorkflowClientBuilder
except ModuleNotFoundError:
    from ._dummyClient import WorkflowClient, WorkflowClientBuilder

from __future__ import annotations

from ...model.workflow import WorkflowExecutionPresenter, WorkflowParametersPresenter
from ...view.workflow import WorkflowExecutionView
from .compute import WorkflowComputeController
from .inputData import WorkflowInputDataController
from .outputData import WorkflowOutputDataController


class WorkflowExecutionController:

    def __init__(self, parametersPresenter: WorkflowParametersPresenter,
                 executionPresenter: WorkflowExecutionPresenter,
                 view: WorkflowExecutionView) -> None:
        self._executionPresenter = executionPresenter
        self._view = view
        self._inputDataController = WorkflowInputDataController.createInstance(
            parametersPresenter, view.inputDataView)
        self._computeController = WorkflowComputeController.createInstance(
            parametersPresenter, view.computeView)
        self._outputDataController = WorkflowOutputDataController.createInstance(
            parametersPresenter, view.outputDataView)

    @classmethod
    def createInstance(cls, parametersPresenter: WorkflowParametersPresenter,
                       executionPresenter: WorkflowExecutionPresenter,
                       view: WorkflowExecutionView) -> WorkflowExecutionController:
        controller = cls(parametersPresenter, executionPresenter, view)
        view.executeButton.clicked.connect(controller._execute)
        return controller

    def _execute(self) -> None:
        flowLabel = self._view.labelLineEdit.text()
        self._executionPresenter.runFlow(flowLabel=flowLabel)

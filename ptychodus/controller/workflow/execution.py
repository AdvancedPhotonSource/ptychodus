from __future__ import annotations
import logging

from PyQt5.QtCore import QAbstractItemModel

from ...model.workflow import WorkflowExecutionPresenter, WorkflowParametersPresenter
from ...view.widgets import ExceptionDialog
from ...view.workflow import WorkflowExecutionView
from .compute import WorkflowComputeController
from .inputData import WorkflowInputDataController
from .outputData import WorkflowOutputDataController

logger = logging.getLogger(__name__)


class WorkflowExecutionController:
    def __init__(
        self,
        parametersPresenter: WorkflowParametersPresenter,
        executionPresenter: WorkflowExecutionPresenter,
        view: WorkflowExecutionView,
    ) -> None:
        self._executionPresenter = executionPresenter
        self._view = view
        self._inputDataController = WorkflowInputDataController.createInstance(
            parametersPresenter, view.inputDataView
        )
        self._computeController = WorkflowComputeController.createInstance(
            parametersPresenter, view.computeView
        )
        self._outputDataController = WorkflowOutputDataController.createInstance(
            parametersPresenter, view.outputDataView
        )

    @classmethod
    def createInstance(
        cls,
        parametersPresenter: WorkflowParametersPresenter,
        executionPresenter: WorkflowExecutionPresenter,
        view: WorkflowExecutionView,
        productItemModel: QAbstractItemModel,
    ) -> WorkflowExecutionController:
        controller = cls(parametersPresenter, executionPresenter, view)
        view.productComboBox.setModel(productItemModel)
        view.executeButton.clicked.connect(controller._execute)
        return controller

    def _execute(self) -> None:
        inputProductIndex = self._view.productComboBox.currentIndex()

        if inputProductIndex < 0:
            logger.debug('No current index!')
            return

        try:
            self._executionPresenter.runFlow(inputProductIndex)
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.showException('Reconstructor', err)

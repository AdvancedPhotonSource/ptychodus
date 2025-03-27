from __future__ import annotations
import logging

from PyQt5.QtCore import QAbstractItemModel

from ...model.workflow import WorkflowExecutionPresenter, WorkflowParametersPresenter
from ...view.widgets import ExceptionDialog
from ...view.workflow import WorkflowExecutionView
from .compute import WorkflowComputeController
from .input_data import WorkflowInputDataController
from .output_data import WorkflowOutputDataController

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
        self._inputDataController = WorkflowInputDataController.create_instance(
            parametersPresenter, view.input_data_view
        )
        self._computeController = WorkflowComputeController.create_instance(
            parametersPresenter, view.compute_view
        )
        self._outputDataController = WorkflowOutputDataController.create_instance(
            parametersPresenter, view.output_data_view
        )

    @classmethod
    def create_instance(
        cls,
        parametersPresenter: WorkflowParametersPresenter,
        executionPresenter: WorkflowExecutionPresenter,
        view: WorkflowExecutionView,
        productItemModel: QAbstractItemModel,
    ) -> WorkflowExecutionController:
        controller = cls(parametersPresenter, executionPresenter, view)
        view.product_combo_box.setModel(productItemModel)
        view.execute_button.clicked.connect(controller._execute)
        return controller

    def _execute(self) -> None:
        inputProductIndex = self._view.product_combo_box.currentIndex()

        if inputProductIndex < 0:
            logger.debug('No current index!')
            return

        try:
            self._executionPresenter.runFlow(inputProductIndex)
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.show_exception('Reconstructor', err)

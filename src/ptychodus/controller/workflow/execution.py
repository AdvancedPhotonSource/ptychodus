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
        parameters_presenter: WorkflowParametersPresenter,
        execution_presenter: WorkflowExecutionPresenter,
        view: WorkflowExecutionView,
        product_item_model: QAbstractItemModel,
    ) -> None:
        self._execution_presenter = execution_presenter
        self._view = view
        self._input_data_controller = WorkflowInputDataController.create_instance(
            parameters_presenter, view.input_data_view
        )
        self._compute_controller = WorkflowComputeController.create_instance(
            parameters_presenter, view.compute_view
        )
        self._output_data_controller = WorkflowOutputDataController.create_instance(
            parameters_presenter, view.output_data_view
        )

        view.product_combo_box.setModel(product_item_model)
        view.execute_button.clicked.connect(self._execute)

    def _execute(self) -> None:
        input_product_index = self._view.product_combo_box.currentIndex()

        if input_product_index < 0:
            logger.debug('No current index!')
            return

        try:
            self._execution_presenter.run_flow(input_product_index)
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.show_exception('Reconstructor', err)

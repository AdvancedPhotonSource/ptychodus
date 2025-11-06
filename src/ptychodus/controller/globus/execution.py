from __future__ import annotations
import logging

from PyQt5.QtCore import QAbstractItemModel

from ...model.globus import GlobusExecutionPresenter, GlobusParametersPresenter
from ...view.widgets import ExceptionDialog
from ...view.globus import GlobusExecutionView
from .compute import GlobusComputeController
from .input_data import GlobusInputDataController
from .output_data import GlobusOutputDataController

logger = logging.getLogger(__name__)


class GlobusExecutionController:
    def __init__(
        self,
        parameters_presenter: GlobusParametersPresenter,
        execution_presenter: GlobusExecutionPresenter,
        view: GlobusExecutionView,
        product_item_model: QAbstractItemModel,
    ) -> None:
        self._execution_presenter = execution_presenter
        self._view = view
        self._input_data_controller = GlobusInputDataController.create_instance(
            parameters_presenter, view.input_data_view
        )
        self._compute_controller = GlobusComputeController.create_instance(
            parameters_presenter, view.compute_view
        )
        self._output_data_controller = GlobusOutputDataController.create_instance(
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

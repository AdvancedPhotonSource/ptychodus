from PyQt5.QtCore import QAbstractItemModel, QTimer
from PyQt5.QtWidgets import QTableView

from ...model.workflow import (
    WorkflowAuthorizationPresenter,
    WorkflowExecutionPresenter,
    WorkflowParametersPresenter,
    WorkflowStatusPresenter,
)
from ...view.workflow import WorkflowParametersView
from .authorization import WorkflowAuthorizationController
from .execution import WorkflowExecutionController
from .status import WorkflowStatusController


class WorkflowController:
    def __init__(
        self,
        parameters_presenter: WorkflowParametersPresenter,
        authorization_presenter: WorkflowAuthorizationPresenter,
        status_presenter: WorkflowStatusPresenter,
        execution_presenter: WorkflowExecutionPresenter,
        parameters_view: WorkflowParametersView,
        table_view: QTableView,
        product_item_model: QAbstractItemModel,
    ) -> None:
        self._parameters_presenter = parameters_presenter
        self._authorization_presenter = authorization_presenter
        self._execution_presenter = execution_presenter
        self._parameters_view = parameters_view
        self._authorization_controller = WorkflowAuthorizationController(
            authorization_presenter, parameters_view.authorization_dialog
        )
        self._status_controller = WorkflowStatusController(
            status_presenter, parameters_view.status_view, table_view
        )
        self._execution_controller = WorkflowExecutionController(
            parameters_presenter,
            execution_presenter,
            parameters_view.execution_view,
            product_item_model,
        )
        self._timer = QTimer()
        self._timer.timeout.connect(self._process_events)
        self._timer.start(5 * 1000)  # TODO customize

    def _process_events(self) -> None:
        self._authorization_controller.start_authorization_if_needed()
        self._status_controller.refresh_table_view()

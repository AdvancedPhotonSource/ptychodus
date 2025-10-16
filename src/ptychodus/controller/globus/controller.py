from PyQt5.QtCore import QAbstractItemModel, QTimer
from PyQt5.QtWidgets import QTableView

from ...model.globus import (
    GlobusAuthorizationPresenter,
    GlobusExecutionPresenter,
    GlobusParametersPresenter,
    GlobusStatusPresenter,
)
from ...view.globus import GlobusParametersView
from .authorization import GlobusAuthorizationController
from .execution import GlobusExecutionController
from .status import GlobusStatusController


class GlobusController:
    def __init__(
        self,
        parameters_presenter: GlobusParametersPresenter,
        authorization_presenter: GlobusAuthorizationPresenter,
        status_presenter: GlobusStatusPresenter,
        execution_presenter: GlobusExecutionPresenter,
        parameters_view: GlobusParametersView,
        table_view: QTableView,
        product_item_model: QAbstractItemModel,
    ) -> None:
        self._parameters_presenter = parameters_presenter
        self._authorization_presenter = authorization_presenter
        self._execution_presenter = execution_presenter
        self._parameters_view = parameters_view
        self._authorization_controller = GlobusAuthorizationController(
            authorization_presenter, parameters_view.authorization_dialog
        )
        self._status_controller = GlobusStatusController(
            status_presenter, parameters_view.status_view, table_view
        )
        self._execution_controller = GlobusExecutionController(
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

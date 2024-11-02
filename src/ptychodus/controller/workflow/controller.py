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
        parametersPresenter: WorkflowParametersPresenter,
        authorizationPresenter: WorkflowAuthorizationPresenter,
        statusPresenter: WorkflowStatusPresenter,
        executionPresenter: WorkflowExecutionPresenter,
        parametersView: WorkflowParametersView,
        tableView: QTableView,
        productItemModel: QAbstractItemModel,
    ) -> None:
        self._parametersPresenter = parametersPresenter
        self._authorizationPresenter = authorizationPresenter
        self._executionPresenter = executionPresenter
        self._parametersView = parametersView
        self._authorizationController = WorkflowAuthorizationController.createInstance(
            authorizationPresenter, parametersView.authorizationDialog
        )
        self._statusController = WorkflowStatusController(
            statusPresenter, parametersView.statusView, tableView
        )
        self._executionController = WorkflowExecutionController.createInstance(
            parametersPresenter,
            executionPresenter,
            parametersView.executionView,
            productItemModel,
        )
        self._timer = QTimer()
        self._timer.timeout.connect(self._processEvents)
        self._timer.start(5 * 1000)  # TODO customize

    def _processEvents(self) -> None:
        self._authorizationController.startAuthorizationIfNeeded()
        self._statusController.refreshTableView()

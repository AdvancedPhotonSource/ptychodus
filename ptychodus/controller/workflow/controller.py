from __future__ import annotations

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QTableView

from ...model.workflow import (WorkflowAuthorizationPresenter, WorkflowExecutionPresenter,
                               WorkflowParametersPresenter, WorkflowStatusPresenter)
from ...view import WorkflowParametersView
from .authorization import WorkflowAuthorizationController
from .execution import WorkflowExecutionController
from .status import WorkflowStatusController


class WorkflowController:

    def __init__(self, parametersPresenter: WorkflowParametersPresenter,
                 authorizationPresenter: WorkflowAuthorizationPresenter,
                 statusPresenter: WorkflowStatusPresenter,
                 executionPresenter: WorkflowExecutionPresenter,
                 parametersView: WorkflowParametersView, tableView: QTableView) -> None:
        self._parametersPresenter = parametersPresenter
        self._authorizationPresenter = authorizationPresenter
        self._executionPresenter = executionPresenter
        self._parametersView = parametersView
        self._authorizationController = WorkflowAuthorizationController.createInstance(
            authorizationPresenter, parametersView.authorizationDialog)
        self._statusController = WorkflowStatusController.createInstance(
            statusPresenter, parametersView.statusView, tableView)
        self._executionController = WorkflowExecutionController.createInstance(
            parametersPresenter, executionPresenter, parametersView.executionView)
        self._timer = QTimer()

    @classmethod
    def createInstance(cls, parametersPresenter: WorkflowParametersPresenter,
                       authorizationPresenter: WorkflowAuthorizationPresenter,
                       statusPresenter: WorkflowStatusPresenter,
                       executionPresenter: WorkflowExecutionPresenter,
                       parametersView: WorkflowParametersView,
                       tableView: QTableView) -> WorkflowController:
        controller = cls(parametersPresenter, authorizationPresenter, statusPresenter,
                         executionPresenter, parametersView, tableView)

        controller._timer.timeout.connect(controller._processEvents)
        controller._timer.start(1000)  # TODO customize

        return controller

    def _processEvents(self) -> None:
        self._authorizationController.startAuthorizationIfNeeded()
        self._statusController.refreshTableViewIfNeeded()

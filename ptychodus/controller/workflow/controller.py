from __future__ import annotations
import logging

from PyQt5.QtCore import Qt, QModelIndex, QSortFilterProxyModel
from PyQt5.QtWidgets import QAbstractItemView, QDialog, QTableView
from PyQt5.QtGui import QDesktopServices

from ...api.observer import Observable, Observer
from ...model.workflow import (WorkflowAuthorizationPresenter, WorkflowExecutionPresenter,
                               WorkflowParametersPresenter, WorkflowRun)
from ...view import WorkflowParametersView
from .authorization import WorkflowAuthorizationController
from .compute import WorkflowComputeController
from .inputData import WorkflowInputDataController
from .outputData import WorkflowOutputDataController
from .tableModel import WorkflowTableModel

logger = logging.getLogger(__name__)


class WorkflowController:

    def __init__(self, parametersPresenter: WorkflowParametersPresenter,
                 authorizationPresenter: WorkflowAuthorizationPresenter,
                 executionPresenter: WorkflowExecutionPresenter,
                 parametersView: WorkflowParametersView, tableView: QTableView) -> None:
        self._parametersPresenter = parametersPresenter
        self._authorizationPresenter = authorizationPresenter
        self._executionPresenter = executionPresenter
        self._parametersView = parametersView
        self._authorizationController = WorkflowAuthorizationController.createInstance(
            authorizationPresenter, parametersView.authorizationDialog)
        self._inputDataController = WorkflowInputDataController.createInstance(
            parametersPresenter, parametersView.executionView.inputDataView)
        self._computeController = WorkflowComputeController.createInstance(
            parametersPresenter, parametersView.executionView.computeView)
        self._outputDataController = WorkflowOutputDataController.createInstance(
            parametersPresenter, parametersView.executionView.outputDataView)
        self._tableView = tableView
        self._tableModel = WorkflowTableModel(executionPresenter)
        self._proxyModel = QSortFilterProxyModel()

    @classmethod
    def createInstance(cls, parametersPresenter: WorkflowParametersPresenter,
                       authorizationPresenter: WorkflowAuthorizationPresenter,
                       executionPresenter: WorkflowExecutionPresenter,
                       parametersView: WorkflowParametersView,
                       tableView: QTableView) -> WorkflowController:
        controller = cls(parametersPresenter, authorizationPresenter, executionPresenter,
                         parametersView, tableView)

        controller._proxyModel.setSourceModel(controller._tableModel)
        tableView.setModel(controller._proxyModel)
        tableView.setSortingEnabled(True)
        tableView.setSelectionBehavior(QAbstractItemView.SelectRows)
        tableView.clicked.connect(controller._handleTableViewClick)

        parametersView.executionView.executeButton.clicked.connect(controller._execute)
        parametersView.statusView.refreshButton.clicked.connect(controller._refreshStatus)

        return controller

    def _handleTableViewClick(self, index: QModelIndex) -> None:
        if index.column() == 5:
            url = index.data(Qt.UserRole)
            logger.debug(f'Opening URL: \"{url.toString()}\"')
            QDesktopServices.openUrl(url)

    def _refreshStatus(self) -> None:
        self._tableModel.refresh()

    def _execute(self) -> None:
        label = self._parametersView.executionView.labelLineEdit.text()
        self._executionPresenter.runFlow(label=label)

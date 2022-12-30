from __future__ import annotations
import logging

from PyQt5.QtCore import Qt, QModelIndex, QSortFilterProxyModel, QTimer
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


class WorkflowController(Observer):

    def __init__(self, parametersPresenter: WorkflowParametersPresenter,
                 authorizationPresenter: WorkflowAuthorizationPresenter,
                 executionPresenter: WorkflowExecutionPresenter,
                 parametersView: WorkflowParametersView, tableView: QTableView) -> None:
        super().__init__()
        self._parametersPresenter = parametersPresenter
        self._authorizationPresenter = authorizationPresenter
        self._executionPresenter = executionPresenter
        self._parametersView = parametersView
        self._authorizationController = WorkflowAuthorizationController.createInstance(
            authorizationPresenter, parametersView.authorizationDialog)
        self._inputDataController = WorkflowInputDataController.createInstance(
            parametersPresenter, parametersView.inputDataView)
        self._outputDataController = WorkflowOutputDataController.createInstance(
            parametersPresenter, parametersView.outputDataView)
        self._computeController = WorkflowComputeController.createInstance(
            parametersPresenter, parametersView.computeView)
        self._tableView = tableView
        self._tableModel = WorkflowTableModel(executionPresenter)
        self._proxyModel = QSortFilterProxyModel()
        self._statusRefreshTimer = QTimer()

    @classmethod
    def createInstance(cls, parametersPresenter: WorkflowParametersPresenter,
                       authorizationPresenter: WorkflowAuthorizationPresenter,
                       executionPresenter: WorkflowExecutionPresenter,
                       parametersView: WorkflowParametersView,
                       tableView: QTableView) -> WorkflowController:
        controller = cls(parametersPresenter, authorizationPresenter, executionPresenter,
                         parametersView, tableView)
        parametersPresenter.addObserver(controller)

        controller._proxyModel.setSourceModel(controller._tableModel)
        tableView.setModel(controller._proxyModel)
        tableView.setSortingEnabled(True)
        tableView.setSelectionBehavior(QAbstractItemView.SelectRows)
        tableView.clicked.connect(controller._handleTableViewClick)

        parametersView.statusView.refreshIntervalSpinBox.setRange(1, 600)
        parametersView.statusView.refreshIntervalSpinBox.valueChanged.connect(
            parametersPresenter.setStatusRefreshIntervalInSeconds)
        parametersView.executeButton.clicked.connect(controller._execute)

        controller._statusRefreshTimer.setSingleShot(True)
        controller._statusRefreshTimer.timeout.connect(controller._refreshStatus)
        controller._syncModelToView()

        return controller

    def _handleTableViewClick(self, index: QModelIndex) -> None:
        if index.column() == 5:
            url = index.data(Qt.UserRole)
            logger.debug(f'Opening URL: \"{url.toString()}\"')
            QDesktopServices.openUrl(url)

    def _restartStatusRefreshTimer(self) -> None:
        seconds = 1000 * self._parametersPresenter.getStatusRefreshIntervalInSeconds()

        if seconds > 0:
            self._statusRefreshTimer.start(seconds)  # FIXME status isn't working
        else:
            # FIXME can't restart
            self._statusRefreshTimer.stop()

    def _refreshStatus(self) -> None:
        self._tableModel.refresh()
        self._restartStatusRefreshTimer()

    def _execute(self) -> None:
        self._executionPresenter.runFlow(label='Ptychodus')  # FIXME label

    def _syncModelToView(self) -> None:
        intervalInSeconds = self._parametersPresenter.getStatusRefreshIntervalInSeconds()
        self._parametersView.statusView.refreshIntervalSpinBox.setValue(intervalInSeconds)

    def update(self, observable: Observable) -> None:
        if observable is self._parametersPresenter:
            self._syncModelToView()

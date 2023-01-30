from __future__ import annotations
import logging

from PyQt5.QtCore import Qt, QModelIndex, QSortFilterProxyModel, QTimer
from PyQt5.QtWidgets import QAbstractItemView, QTableView
from PyQt5.QtGui import QDesktopServices

from ...model.workflow import WorkflowStatusPresenter
from ...view import WorkflowStatusView
from .tableModel import WorkflowTableModel

logger = logging.getLogger(__name__)


class WorkflowStatusController:

    def __init__(self, presenter: WorkflowStatusPresenter, view: WorkflowStatusView,
                 tableView: QTableView) -> None:
        self._presenter = presenter
        self._view = view
        self._tableView = tableView
        self._tableModel = WorkflowTableModel(presenter)
        self._proxyModel = QSortFilterProxyModel()
        self._timer = QTimer()

    @classmethod
    def createInstance(cls, presenter: WorkflowStatusPresenter, view: WorkflowStatusView,
                       tableView: QTableView) -> WorkflowStatusController:
        controller = cls(presenter, view, tableView)

        controller._proxyModel.setSourceModel(controller._tableModel)
        tableView.setModel(controller._proxyModel)
        tableView.setSortingEnabled(True)
        tableView.setSelectionBehavior(QAbstractItemView.SelectRows)
        tableView.clicked.connect(controller._handleTableViewClick)

        controller._timer.timeout.connect(presenter.refreshStatus)
        view.autoRefreshCheckBox.toggled.connect(controller._autoRefreshStatus)
        view.autoRefreshSpinBox.valueChanged.connect(presenter.setRefreshIntervalInSeconds)
        view.refreshButton.clicked.connect(presenter.refreshStatus)

        controller._syncModelToView()

        return controller

    def _handleTableViewClick(self, index: QModelIndex) -> None:
        if index.column() == 5:
            url = index.data(Qt.UserRole)
            logger.debug(f'Opening URL: \"{url.toString()}\"')
            QDesktopServices.openUrl(url)

    def _autoRefreshStatus(self) -> None:
        if self._view.autoRefreshCheckBox.isChecked():
            self._timer.start(1000 * self._presenter.getRefreshIntervalInSeconds())
            self._view.autoRefreshSpinBox.setEnabled(False)
            self._view.refreshButton.setEnabled(False)
        else:
            self._timer.stop()
            self._view.autoRefreshSpinBox.setEnabled(True)
            self._view.refreshButton.setEnabled(True)

    def refreshTableViewIfChanged(self) -> None:  # FIXME only reset if changed
        self._tableModel.beginResetModel()
        self._tableModel.endResetModel()

    def _syncModelToView(self) -> None:
        refreshIntervalLimitsInSeconds = self._presenter.getRefreshIntervalLimitsInSeconds()

        self._view.autoRefreshSpinBox.blockSignals(True)
        self._view.autoRefreshSpinBox.setRange(refreshIntervalLimitsInSeconds.lower,
                                               refreshIntervalLimitsInSeconds.upper)
        self._view.autoRefreshSpinBox.setValue(self._presenter.getRefreshIntervalInSeconds())
        self._view.autoRefreshSpinBox.blockSignals(False)

import logging

from PyQt5.QtCore import Qt, QModelIndex, QSortFilterProxyModel, QTimer
from PyQt5.QtWidgets import QAbstractItemView, QTableView
from PyQt5.QtGui import QDesktopServices

from ptychodus.api.observer import Observable, Observer

from ...model.workflow import WorkflowStatusPresenter
from ...view.workflow import WorkflowStatusView
from .tableModel import WorkflowTableModel

logger = logging.getLogger(__name__)


class WorkflowStatusController(Observer):
    def __init__(
        self,
        presenter: WorkflowStatusPresenter,
        view: WorkflowStatusView,
        tableView: QTableView,
    ) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view
        self._tableView = tableView
        self._tableModel = WorkflowTableModel(presenter)
        self._proxyModel = QSortFilterProxyModel()
        self._proxyModel.setSourceModel(self._tableModel)
        self._timer = QTimer()
        self._timer.timeout.connect(presenter.refreshStatus)

        tableView.setModel(self._proxyModel)
        tableView.setSortingEnabled(True)
        tableView.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        tableView.clicked.connect(self._handleTableViewClick)

        view.autoRefreshCheckBox.toggled.connect(self._autoRefreshStatus)
        view.autoRefreshSpinBox.valueChanged.connect(presenter.setRefreshIntervalInSeconds)
        view.refreshButton.clicked.connect(presenter.refreshStatus)

        self._syncModelToView()
        presenter.addObserver(self)

    def _handleTableViewClick(self, index: QModelIndex) -> None:
        if index.column() == 5:
            url = index.data(Qt.ItemDataRole.UserRole)
            logger.info(f'Opening URL: "{url.toString()}"')
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

    def refreshTableView(self) -> None:
        # TODO only reset if changed
        self._tableModel.beginResetModel()
        self._tableModel.endResetModel()

    def _syncModelToView(self) -> None:
        refreshIntervalLimitsInSeconds = self._presenter.getRefreshIntervalLimitsInSeconds()

        self._view.autoRefreshSpinBox.blockSignals(True)
        self._view.autoRefreshSpinBox.setRange(
            refreshIntervalLimitsInSeconds.lower, refreshIntervalLimitsInSeconds.upper
        )
        self._view.autoRefreshSpinBox.setValue(self._presenter.getRefreshIntervalInSeconds())
        self._view.autoRefreshSpinBox.blockSignals(False)

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()

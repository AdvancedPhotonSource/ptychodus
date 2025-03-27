import logging

from PyQt5.QtCore import Qt, QModelIndex, QSortFilterProxyModel, QTimer
from PyQt5.QtWidgets import QAbstractItemView, QTableView
from PyQt5.QtGui import QDesktopServices

from ptychodus.api.observer import Observable, Observer

from ...model.workflow import WorkflowStatusPresenter
from ...view.workflow import WorkflowStatusView
from .table_model import WorkflowTableModel

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

        view.auto_refresh_check_box.toggled.connect(self._autoRefreshStatus)
        view.auto_refresh_spin_box.valueChanged.connect(presenter.setRefreshIntervalInSeconds)
        view.refresh_button.clicked.connect(presenter.refreshStatus)

        self._sync_model_to_view()
        presenter.add_observer(self)

    def _handleTableViewClick(self, index: QModelIndex) -> None:
        if index.column() == 5:
            url = index.data(Qt.ItemDataRole.UserRole)
            logger.info(f'Opening URL: "{url.toString()}"')
            QDesktopServices.openUrl(url)

    def _autoRefreshStatus(self) -> None:
        if self._view.auto_refresh_check_box.isChecked():
            self._timer.start(1000 * self._presenter.getRefreshIntervalInSeconds())
            self._view.auto_refresh_spin_box.setEnabled(False)
            self._view.refresh_button.setEnabled(False)
        else:
            self._timer.stop()
            self._view.auto_refresh_spin_box.setEnabled(True)
            self._view.refresh_button.setEnabled(True)

    def refreshTableView(self) -> None:
        # TODO only reset if changed
        self._tableModel.beginResetModel()
        self._tableModel.endResetModel()

    def _sync_model_to_view(self) -> None:
        refreshIntervalLimitsInSeconds = self._presenter.getRefreshIntervalLimitsInSeconds()

        self._view.auto_refresh_spin_box.blockSignals(True)
        self._view.auto_refresh_spin_box.setRange(
            refreshIntervalLimitsInSeconds.lower, refreshIntervalLimitsInSeconds.upper
        )
        self._view.auto_refresh_spin_box.setValue(self._presenter.getRefreshIntervalInSeconds())
        self._view.auto_refresh_spin_box.blockSignals(False)

    def _update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._sync_model_to_view()

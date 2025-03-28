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
        table_view: QTableView,
    ) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view
        self._table_view = table_view
        self._table_model = WorkflowTableModel(presenter)
        self._proxy_model = QSortFilterProxyModel()
        self._proxy_model.setSourceModel(self._table_model)
        self._timer = QTimer()
        self._timer.timeout.connect(presenter.refresh_status)

        table_view.setModel(self._proxy_model)
        table_view.setSortingEnabled(True)
        table_view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table_view.clicked.connect(self._handle_table_view_clicked)

        view.auto_refresh_check_box.toggled.connect(self._auto_refresh_status)
        view.auto_refresh_spin_box.valueChanged.connect(presenter.set_refresh_interval_s)
        view.refresh_button.clicked.connect(presenter.refresh_status)

        self._sync_model_to_view()
        presenter.add_observer(self)

    def _handle_table_view_clicked(self, index: QModelIndex) -> None:
        if index.column() == 5:
            url = index.data(Qt.ItemDataRole.UserRole)
            logger.info(f'Opening URL: "{url.toString()}"')
            QDesktopServices.openUrl(url)

    def _auto_refresh_status(self) -> None:
        if self._view.auto_refresh_check_box.isChecked():
            self._timer.start(1000 * self._presenter.get_refresh_interval_s())
            self._view.auto_refresh_spin_box.setEnabled(False)
            self._view.refresh_button.setEnabled(False)
        else:
            self._timer.stop()
            self._view.auto_refresh_spin_box.setEnabled(True)
            self._view.refresh_button.setEnabled(True)

    def refresh_table_view(self) -> None:
        # TODO only reset if changed
        self._table_model.beginResetModel()
        self._table_model.endResetModel()

    def _sync_model_to_view(self) -> None:
        refresh_interval_limits_s = self._presenter.get_refresh_interval_limits_s()

        self._view.auto_refresh_spin_box.blockSignals(True)
        self._view.auto_refresh_spin_box.setRange(
            refresh_interval_limits_s.lower, refresh_interval_limits_s.upper
        )
        self._view.auto_refresh_spin_box.setValue(self._presenter.get_refresh_interval_s())
        self._view.auto_refresh_spin_box.blockSignals(False)

    def _update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._sync_model_to_view()

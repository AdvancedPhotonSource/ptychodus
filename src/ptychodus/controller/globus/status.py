from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QPushButton,
    QWidget,
)

from ptychodus.api.parametric import IntegerParameter

from ...controller.parametric import SpinBoxParameterViewController
from ...model.globus import GlobusStatusRepository
from ..parametric import ParameterViewController


class GlobusStatusViewController(ParameterViewController):
    def __init__(
        self,
        status_refresh_interval_s: IntegerParameter,
        status_repository: GlobusStatusRepository,
    ) -> None:
        super().__init__()
        self._status_refresh_interval_s = status_refresh_interval_s
        self._status_repository = status_repository
        self._timer = QTimer()
        # FIXME (to model) self._timer.timeout.connect(status_repository.refresh_status)

        self._auto_refresh_check_box = QCheckBox('Auto Refresh [sec]:')
        self._auto_refresh_check_box.toggled.connect(self._auto_refresh_status)
        self._status_refresh_interval_view_controller = SpinBoxParameterViewController(
            status_refresh_interval_s
        )
        self._refresh_button = QPushButton('Refresh')
        # FIXME self._refresh_button.clicked.connect(status_repository.refresh_status)

        layout = QFormLayout()
        layout.addRow(
            self._auto_refresh_check_box,
            self._status_refresh_interval_view_controller.get_widget(),
        )
        layout.addRow(self._refresh_button)

        self._widget = QGroupBox('Status')
        self._widget.setLayout(layout)

        self._auto_refresh_status()

    def get_widget(self) -> QWidget:
        return self._widget

    def _auto_refresh_status(self) -> None:
        status_refresh_interval_widget = self._status_refresh_interval_view_controller.get_widget()

        if self._auto_refresh_check_box.isChecked():
            self._timer.start(1000 * self._status_refresh_interval_s.get_value())
            status_refresh_interval_widget.setEnabled(False)
            self._refresh_button.setEnabled(False)
        else:
            self._timer.stop()
            status_refresh_interval_widget.setEnabled(True)
            self._refresh_button.setEnabled(True)

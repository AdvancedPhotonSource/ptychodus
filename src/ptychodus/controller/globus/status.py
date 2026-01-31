from PyQt5.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QPushButton,
    QWidget,
)

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.parametric import BooleanParameter, IntegerParameter

from ...controller.parametric import SpinBoxParameterViewController
from ...model.globus import GlobusStatusRepository
from ..parametric import CheckBoxParameterViewController, ParameterViewController


class GlobusStatusViewController(ParameterViewController, Observer):
    def __init__(
        self,
        status_auto_refresh: BooleanParameter,
        status_refresh_interval_s: IntegerParameter,
        status_repository: GlobusStatusRepository,
    ) -> None:
        super().__init__()
        self._status_auto_refresh = status_auto_refresh
        self._status_refresh_interval_s = status_refresh_interval_s
        self._status_repository = status_repository

        self._auto_refresh_view_controller = CheckBoxParameterViewController(
            status_auto_refresh, 'Auto Refresh [sec]:'
        )
        self._status_refresh_interval_view_controller = SpinBoxParameterViewController(
            status_refresh_interval_s
        )
        self._refresh_button = QPushButton('Refresh')
        # FIXME self._refresh_button.clicked.connect(status_repository.refresh_status)

        layout = QFormLayout()
        layout.addRow(
            self._auto_refresh_view_controller.get_widget(),
            self._status_refresh_interval_view_controller.get_widget(),
        )
        layout.addRow(self._refresh_button)

        self._widget = QGroupBox('Status')
        self._widget.setLayout(layout)

        status_auto_refresh.add_observer(self)
        self._update_enabled_widgets()

    def get_widget(self) -> QWidget:
        return self._widget

    def _update_enabled_widgets(self) -> None:
        status_refresh_interval_widget = self._status_refresh_interval_view_controller.get_widget()
        enable = not self._status_auto_refresh.get_value()
        status_refresh_interval_widget.setEnabled(enable)
        self._refresh_button.setEnabled(enable)

    def _update(self, observable: Observable) -> None:
        if observable is self._status_auto_refresh:
            self._update_enabled_widgets()

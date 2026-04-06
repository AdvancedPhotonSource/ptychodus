from pathlib import Path

from PyQt5.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.parametric import BooleanParameter, IntegerParameter, PathParameter

from ..model.genesis import GenesisSettings
from .data import FileDialogFactory
from .parametric import (
    CheckBoxParameterViewController,
    ParameterViewBuilder,
    ParameterViewController,
    SpinBoxParameterViewController,
)


class PathLineEditParameterViewController(ParameterViewController, Observer):
    # FIXME: to ptychodus.controller.parametric

    def __init__(self, parameter: PathParameter, *, tool_tip: str = '') -> None:
        super().__init__()
        self._parameter = parameter
        self._widget = QLineEdit()

        if tool_tip:
            self._widget.setToolTip(tool_tip)

        self.__sync_model_to_view()
        self._widget.editingFinished.connect(self.__sync_view_to_model)
        parameter.add_observer(self)

    def get_widget(self) -> QWidget:
        return self._widget

    def __sync_view_to_model(self) -> None:
        self._parameter.set_value(Path(self._widget.text()))

    def __sync_model_to_view(self) -> None:
        self._widget.setText(str(self._parameter.get_value()))

    def _update(self, observable: Observable) -> None:
        if observable is self._parameter:
            self.__sync_model_to_view()


class GenesisStatusViewController(ParameterViewController, Observer):
    def __init__(
        self,
        status_auto_refresh: BooleanParameter,
        status_refresh_interval_s: IntegerParameter,
    ) -> None:
        super().__init__()
        self._status_auto_refresh = status_auto_refresh
        self._status_refresh_interval_s = status_refresh_interval_s

        self._auto_refresh_view_controller = CheckBoxParameterViewController(
            status_auto_refresh, 'Auto Refresh [sec]:'
        )
        self._status_refresh_interval_view_controller = SpinBoxParameterViewController(
            status_refresh_interval_s
        )
        self._refresh_button = QPushButton('Refresh')

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


class GenesisController:
    def __init__(
        self, settings: GenesisSettings, view: QWidget, file_dialog_factory: FileDialogFactory
    ) -> None:
        self._settings = settings
        self._view = view

        self._status_controller = GenesisStatusViewController(
            settings.status_auto_refresh,
            settings.status_refresh_interval_s,
        )

        view_builder = ParameterViewBuilder(file_dialog_factory)
        view_builder.add_line_edit(settings.api_base_url, 'API Base URL:')

        local_group = 'Local'
        view_builder.add_uuid_line_edit(
            settings.local_collection_id, 'Collection UUID:', group=local_group
        )
        view_builder.add_line_edit(
            settings.local_collection_globus_path,
            'Collection Globus Path:',
            tool_tip='Globus path on the local system where data is stored.',
            group=local_group,
        )
        view_builder.add_directory_chooser(
            settings.local_collection_posix_path,
            'Collection POSIX Path:',
            tool_tip='POSIX path on the local system where data is stored.',
            group=local_group,
        )

        remote_group = 'Remote'
        view_builder.add_uuid_line_edit(
            settings.remote_collection_id, 'Collection UUID:', group=remote_group
        )
        view_builder.add_line_edit(
            settings.remote_collection_globus_path,
            'Collection Globus Path:',
            tool_tip='Globus path on the remote system where data is stored.',
            group=remote_group,
        )
        view_builder.add_directory_chooser(
            settings.remote_collection_posix_path,
            'Collection POSIX Path:',
            tool_tip='POSIX path on the remote system where data is stored.',
            group=remote_group,
        )

        view_builder.add_view_controller_to_bottom(self._status_controller)
        contents = view_builder.build_widget()

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(contents)
        view.setLayout(layout)

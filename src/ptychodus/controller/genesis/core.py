from PyQt5.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.parametric import BooleanParameter, IntegerParameter, StringParameter

from ...model.genesis import GenesisPresenter, GenesisSettings
from ..data import FileDialogFactory
from ..parametric import (
    CheckBoxParameterViewController,
    ParameterViewBuilder,
    ParameterViewController,
    PathLineEditParameterViewController,
    SpinBoxParameterViewController,
)


class ComputeResourceComboBoxViewController(ParameterViewController, Observer):
    def __init__(
        self,
        compute_resource_id: StringParameter,
        facility: StringParameter,
        presenter: GenesisPresenter,
    ) -> None:
        super().__init__()
        self._compute_resource_id = compute_resource_id
        self._facility = facility
        self._presenter = presenter
        self._widget = QComboBox()

        self._repopulate()
        self._sync_model_to_view()

        self._widget.textActivated.connect(self._handle_text_activated)
        compute_resource_id.add_observer(self)
        facility.add_observer(self)

    def get_widget(self) -> QComboBox:
        return self._widget

    def _repopulate(self) -> None:
        self._widget.blockSignals(True)
        self._widget.clear()

        for name in self._presenter.supported_compute_resources():
            self._widget.addItem(name)

        self._widget.blockSignals(False)

    def _handle_text_activated(self, resource_name: str) -> None:
        self._compute_resource_id.set_value(
            self._presenter.map_compute_resource_name_to_id(resource_name)
        )

    def _sync_model_to_view(self) -> None:
        name = self._presenter.map_compute_resource_id_to_name(
            self._compute_resource_id.get_value()
        )
        self._widget.setCurrentText(name)

    def _update(self, observable: Observable) -> None:
        if observable is self._facility:
            self._repopulate()
            self._sync_model_to_view()
        elif observable is self._compute_resource_id:
            self._sync_model_to_view()


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
        self,
        settings: GenesisSettings,
        presenter: GenesisPresenter,
        view: QWidget,
        file_dialog_factory: FileDialogFactory,
    ) -> None:
        self._settings = settings
        self._view = view
        self._presenter = presenter

        self._status_controller = GenesisStatusViewController(
            settings.status_auto_refresh,
            settings.status_refresh_interval_s,
        )
        self._local_collection_posix_path_controller = PathLineEditParameterViewController(
            settings.local_collection_posix_path,
            tool_tip='POSIX path on the local system where data is stored.',
        )
        self._remote_collection_posix_path_controller = PathLineEditParameterViewController(
            settings.remote_collection_posix_path,
            tool_tip='POSIX path on the remote system where data is stored.',
        )

        self._compute_resource_controller = ComputeResourceComboBoxViewController(
            settings.compute_resource_id,
            settings.facility,
            presenter,
        )

        view_builder = ParameterViewBuilder(file_dialog_factory)

        genesis_group = 'American Science Cloud'
        view_builder.add_combo_box(
            settings.facility,
            presenter.supported_facilities(),
            'IRI Facility:',
            group=genesis_group,
        )
        view_builder.add_view_controller(
            self._compute_resource_controller,
            'IRI Compute Resource:',
            group=genesis_group,
        )
        view_builder.add_combo_box(
            settings.globus_transfer_provider,
            presenter.supported_transfer_clients(),
            'Globus Transfer Provider:',
            group=genesis_group,
        )
        # FIXME add button to call presenter.apply_facility_defaults() to fill in the default values for the selected facility

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
        view_builder.add_view_controller(
            self._local_collection_posix_path_controller,
            'Collection POSIX Path:',
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
        view_builder.add_view_controller(
            self._remote_collection_posix_path_controller,
            'Collection POSIX Path:',
            group=remote_group,
        )

        view_builder.add_view_controller_to_bottom(self._status_controller)
        contents = view_builder.build_widget()

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(contents)
        view.setLayout(layout)

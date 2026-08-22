from PyQt5.QtCore import QModelIndex, QSortFilterProxyModel
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from ptychodus.api.observer import Observable, Observer, SequenceObserver
from ptychodus.api.parameters import StringParameter

from ...model.genesis import (
    GenesisPresenter,
    GenesisSettings,
    GenesisStatus,
    GenesisStatusRepository,
)
from ..data import FileDialogFactory
from ..parameters import (
    LineEditParameterViewController,
    ParameterViewBuilder,
    ParameterViewController,
    PathLineEditParameterViewController,
)
from .table_model import GenesisStatusTableModel


class AccountComboBoxViewController(ParameterViewController, Observer):
    def __init__(
        self,
        account: StringParameter,
        facility: StringParameter,
        presenter: GenesisPresenter,
    ) -> None:
        super().__init__()
        self._account = account
        self._facility = facility
        self._presenter = presenter
        self._widget = QComboBox()

        self._repopulate()
        self._sync_model_to_view()

        self._widget.textActivated.connect(self._handle_text_activated)
        account.add_observer(self)
        facility.add_observer(self)

    def get_widget(self) -> QComboBox:
        return self._widget

    def _repopulate(self) -> None:
        self._widget.blockSignals(True)
        self._widget.clear()

        for name in self._presenter.supported_projects():
            self._widget.addItem(name)

        self._widget.blockSignals(False)

    def _handle_text_activated(self, account_name: str) -> None:
        self._account.set_value(self._presenter.map_project_name_to_id(account_name))

    def _sync_model_to_view(self) -> None:
        name = self._presenter.map_project_id_to_name(self._account.get_value())
        self._widget.setCurrentText(name)

    def _update(self, observable: Observable) -> None:
        if observable is self._facility:
            self._repopulate()
            self._sync_model_to_view()
        elif observable is self._account:
            self._sync_model_to_view()


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


class GenesisController(SequenceObserver[GenesisStatus]):
    def __init__(
        self,
        settings: GenesisSettings,
        presenter: GenesisPresenter,
        status_repository: GenesisStatusRepository,
        view: QWidget,
        status_view: QTableView,
        file_dialog_factory: FileDialogFactory,
    ) -> None:
        super().__init__()
        self._settings = settings
        self._presenter = presenter
        self._view = view

        self._status_table_model = GenesisStatusTableModel(status_repository)
        self._status_proxy_model = QSortFilterProxyModel()
        self._status_proxy_model.setSourceModel(self._status_table_model)

        status_view.setModel(self._status_proxy_model)
        status_view.setSortingEnabled(True)
        status_view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        header = status_view.horizontalHeader()
        header.setSectionResizeMode(header.ResizeMode.ResizeToContents)

        status_repository.add_observer(self)

        self._local_collection_posix_path_view_controller = PathLineEditParameterViewController(
            settings.local_collection_posix_path,
            tool_tip='POSIX path on the local system where data is stored.',
        )
        self._remote_collection_posix_path_view_controller = PathLineEditParameterViewController(
            settings.remote_collection_posix_path,
            tool_tip='POSIX path on the remote system where data is stored.',
        )

        self._compute_resource_view_controller = ComputeResourceComboBoxViewController(
            settings.compute_resource_id,
            settings.facility,
            presenter,
        )
        if False:  # FIXME
            self._account_view_controller: ParameterViewController = AccountComboBoxViewController(
                settings.account,
                settings.facility,
                presenter,
            )
        else:
            self._account_view_controller = LineEditParameterViewController(
                settings.account,
            )

        view_builder = ParameterViewBuilder(file_dialog_factory)

        genesis_group = 'American Science Cloud'
        view_builder.add_combo_box(
            settings.facility,
            presenter.supported_facilities(),
            'IRI Facility:',
            group=genesis_group,
        )
        view_builder.add_combo_box(
            settings.globus_transfer_provider,
            presenter.supported_transfer_clients(),
            'Globus Transfer Provider:',
            group=genesis_group,
        )
        view_builder.add_spin_box(
            settings.status_refresh_interval_s,
            'Status Refresh Interval [s]:',
            group=genesis_group,
        )

        compute_group = 'Compute'
        view_builder.add_view_controller(
            self._compute_resource_view_controller,
            'IRI Resource:',
            group=compute_group,
        )
        view_builder.add_view_controller(
            self._account_view_controller,
            'Account:',
            group=compute_group,
        )
        view_builder.add_line_edit(
            settings.queue_name,
            'Queue:',
            group=compute_group,
        )
        view_builder.add_integer_line_edit(
            settings.duration_s,
            'Duration [s]:',
            group=compute_group,
        )

        local_group = 'Local Collection'
        view_builder.add_uuid_line_edit(settings.local_collection_id, 'UUID:', group=local_group)
        view_builder.add_line_edit(
            settings.local_collection_globus_path,
            'Globus Path:',
            tool_tip='Globus path on the local system where data is stored.',
            group=local_group,
        )
        view_builder.add_view_controller(
            self._local_collection_posix_path_view_controller,
            'POSIX Path:',
            group=local_group,
        )

        remote_group = 'Remote Collection'
        view_builder.add_uuid_line_edit(settings.remote_collection_id, 'UUID:', group=remote_group)
        view_builder.add_line_edit(
            settings.remote_collection_globus_path,
            'Globus Path:',
            tool_tip='Globus path on the remote system where data is stored.',
            group=remote_group,
        )
        view_builder.add_view_controller(
            self._remote_collection_posix_path_view_controller,
            'POSIX Path:',
            group=remote_group,
        )

        contents = view_builder.build_widget()

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(contents)
        view.setLayout(layout)

    def handle_item_inserted(self, index: int, item: GenesisStatus) -> None:
        self._status_table_model.beginInsertRows(QModelIndex(), index, index)
        self._status_table_model.endInsertRows()

    def handle_item_changed(self, index: int, item: GenesisStatus) -> None:
        top_left = self._status_table_model.index(index, 0)
        bottom_right = self._status_table_model.index(index, self._status_table_model.columnCount())
        self._status_table_model.dataChanged.emit(top_left, bottom_right)

    def handle_item_removed(self, index: int, item: GenesisStatus) -> None:
        self._status_table_model.beginRemoveRows(QModelIndex(), index, index)
        self._status_table_model.endRemoveRows()

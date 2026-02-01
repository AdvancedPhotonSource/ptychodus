from pathlib import Path
import logging

from PyQt5.QtCore import QModelIndex, QSortFilterProxyModel, Qt
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QLineEdit,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from ptychodus.api.observer import Observable, Observer, SequenceObserver
from ptychodus.api.parametric import PathParameter

from ...model.globus import GlobusAuthorizer, GlobusSettings, GlobusStatus, GlobusStatusRepository
from ..data import FileDialogFactory
from ..parametric import ParameterViewBuilder, ParameterViewController
from .authorization import GlobusAuthorizationController
from .status import GlobusStatusViewController
from .table_model import GlobusStatusTableModel

logger = logging.getLogger(__name__)


class PathLineEditParameterViewController(ParameterViewController, Observer):
    # TODO: to ptychodus.controller.parametric

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


class GlobusController(SequenceObserver[GlobusStatus]):
    def __init__(
        self,
        settings: GlobusSettings,
        authorizer: GlobusAuthorizer,
        status_repository: GlobusStatusRepository,
        view: QWidget,
        status_table_view: QTableView,
        file_dialog_factory: FileDialogFactory,
    ) -> None:
        super().__init__()
        self._settings = settings
        self._view = view
        self._auth_controller = GlobusAuthorizationController(authorizer, view)
        self._compute_data_posix_path_controller = PathLineEditParameterViewController(
            settings.compute_data_posix_path,
            tool_tip='POSIX path on the compute endpoint where data will be stored.',
        )
        self._output_data_posix_path_controller = PathLineEditParameterViewController(
            settings.output_data_posix_path,
            tool_tip='POSIX path on the output data endpoint where data will be stored.',
        )
        self._status_controller = GlobusStatusViewController(
            settings.status_auto_refresh, settings.status_refresh_interval_s, status_repository
        )

        self._status_table_model = GlobusStatusTableModel(status_repository)
        self._status_proxy_model = QSortFilterProxyModel()
        self._status_proxy_model.setSourceModel(self._status_table_model)

        status_table_view.setModel(self._status_proxy_model)
        status_table_view.setSortingEnabled(True)
        status_table_view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        status_table_view.clicked.connect(self._handle_table_view_clicked)

        status_repository.add_observer(self)

        # FIXME option for launching Globus in reconstructor view:
        #       workflow_api.get_product(product_index).reconstruct_remote()

        view_builder = ParameterViewBuilder(file_dialog_factory)

        input_data_group = 'Input Data'
        view_builder.add_uuid_line_edit(
            settings.input_data_endpoint_id, 'Endpoint ID:', group=input_data_group
        )
        view_builder.add_line_edit(
            settings.input_data_globus_path, 'Globus Path:', group=input_data_group
        )
        view_builder.add_directory_chooser(
            settings.input_data_posix_path, 'POSIX Path:', group=input_data_group
        )

        compute_group = 'Compute'
        view_builder.add_uuid_line_edit(
            settings.compute_endpoint_id, 'Compute Endpoint ID:', group=compute_group
        )
        view_builder.add_uuid_line_edit(
            settings.compute_data_endpoint_id, 'Data Endpoint ID:', group=compute_group
        )
        view_builder.add_line_edit(
            settings.compute_data_globus_path, 'Data Globus Path:', group=compute_group
        )
        view_builder.add_view_controller(
            self._compute_data_posix_path_controller, 'Data POSIX Path:', group=compute_group
        )

        output_data_group = 'Output Data'
        view_builder.add_uuid_line_edit(
            settings.output_data_endpoint_id, 'Endpoint ID:', group=output_data_group
        )
        view_builder.add_line_edit(
            settings.output_data_globus_path, 'Globus Path:', group=output_data_group
        )
        view_builder.add_view_controller(
            self._output_data_posix_path_controller, 'POSIX Path:', group=output_data_group
        )

        view_builder.add_view_controller_to_bottom(self._status_controller)
        contents = view_builder.build_widget()

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(contents)
        view.setLayout(layout)

    def _handle_table_view_clicked(self, index: QModelIndex) -> None:
        if index.column() == 5:
            url = index.data(Qt.ItemDataRole.UserRole)
            logger.info(f'Opening URL: "{url.toString()}"')
            QDesktopServices.openUrl(url)

    def run_tasks(self) -> None:
        # FIXME remove; modal dialog; use authorizer.needs_authorization
        self._auth_controller.start_authorization_if_needed()

    def handle_item_inserted(self, index: int, item: GlobusStatus) -> None:
        self._status_table_model.beginInsertRows(QModelIndex(), index, index)
        self._status_table_model.endInsertRows()

    def handle_item_changed(self, index: int, item: GlobusStatus) -> None:
        top_left = self._status_table_model.index(index, 0)
        bottom_right = self._status_table_model.index(index, self._status_table_model.columnCount())
        self._status_table_model.dataChanged.emit(top_left, bottom_right)

    def handle_item_removed(self, index: int, item: GlobusStatus) -> None:
        self._status_table_model.beginRemoveRows(QModelIndex(), index, index)
        self._status_table_model.endRemoveRows()

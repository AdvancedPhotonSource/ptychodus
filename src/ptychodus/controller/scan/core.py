from __future__ import annotations
import logging

from PyQt5.QtCore import QModelIndex, QSortFilterProxyModel, QStringListModel
from PyQt5.QtWidgets import QAbstractItemView, QDialog, QMessageBox

from ptychodus.api.observer import SequenceObserver

from ...model.product import ScanAPI, ScanRepository
from ...model.product.scan import ScanRepositoryItem
from ...view.repository import RepositoryTableView
from ...view.scan import ScanPlotView
from ...view.widgets import ComboBoxItemDelegate, ExceptionDialog
from ..data import FileDialogFactory
from ..helpers import (
    connect_current_changed_signal,
    connect_triggered_signal,
    create_brush_for_editable_cell,
)
from .editor_factory import ScanEditorViewControllerFactory
from .table_model import ScanTableModel

logger = logging.getLogger(__name__)


class ScanController(SequenceObserver[ScanRepositoryItem]):
    def __init__(
        self,
        repository: ScanRepository,
        api: ScanAPI,
        view: RepositoryTableView,
        plot_view: ScanPlotView,
        file_dialog_factory: FileDialogFactory,
        *,
        is_developer_mode_enabled: bool,
    ) -> None:
        super().__init__()
        self._repository = repository
        self._api = api
        self._view = view
        self._plot_view = plot_view
        self._file_dialog_factory = file_dialog_factory
        self._table_model = ScanTableModel(repository, api, create_brush_for_editable_cell(view))
        self._table_proxy_model = QSortFilterProxyModel()
        self._editor_factory = ScanEditorViewControllerFactory()

        self._table_proxy_model.setSourceModel(self._table_model)
        self._table_proxy_model.dataChanged.connect(
            lambda top_left, bottom_right, roles: self._redraw_plot()
        )
        repository.add_observer(self)

        builder_list_model = QStringListModel()
        builder_list_model.setStringList([name for name in api.builder_names()])
        builder_item_delegate = ComboBoxItemDelegate(builder_list_model, view.table_view)

        view.table_view.setModel(self._table_proxy_model)
        view.table_view.setSortingEnabled(True)
        view.table_view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        view.table_view.setItemDelegateForColumn(2, builder_item_delegate)
        connect_current_changed_signal(view.table_view, self._update_view)
        self._update_view(QModelIndex(), QModelIndex())

        view.table_view.horizontalHeader().sectionClicked.connect(
            lambda logical_index: self._redraw_plot()
        )

        load_from_file_action = view.button_box.load_menu.addAction('Open File...')
        connect_triggered_signal(load_from_file_action, self._load_current_scan_from_file)

        copy_action = view.button_box.load_menu.addAction('Copy...')
        connect_triggered_signal(copy_action, self._copy_to_current_scan)

        save_to_file_action = view.button_box.save_menu.addAction('Save File...')
        connect_triggered_signal(save_to_file_action, self._save_current_scan_to_file)

        sync_to_settings_action = view.button_box.save_menu.addAction('Sync To Settings')
        connect_triggered_signal(sync_to_settings_action, self._sync_current_scan_to_settings)

        view.copier_dialog.setWindowTitle('Copy Scan')
        view.copier_dialog.source_combo_box.setModel(self._table_model)
        view.copier_dialog.destination_combo_box.setModel(self._table_model)
        view.copier_dialog.finished.connect(self._finish_copying_scan)

        view.button_box.edit_button.clicked.connect(self._edit_current_scan)

        estimate_transform_action = view.button_box.analyze_menu.addAction('Estimate Transform...')
        connect_triggered_signal(estimate_transform_action, self._estimate_transform)
        estimate_transform_action.setEnabled(is_developer_mode_enabled)

    def _get_current_item_index(self) -> int:
        proxy_index = self._view.table_view.currentIndex()

        if proxy_index.isValid():
            model_index = self._table_proxy_model.mapToSource(proxy_index)
            return model_index.row()

        logger.warning('No current index!')
        return -1

    def _load_current_scan_from_file(self) -> None:
        item_index = self._get_current_item_index()

        if item_index < 0:
            return

        file_path, name_filter = self._file_dialog_factory.get_open_file_path(
            self._view,
            'Open Scan',
            name_filters=[nf for nf in self._api.get_open_file_filters()],
            selected_name_filter=self._api.get_open_file_filter(),
        )

        if file_path:
            try:
                self._api.open_scan(item_index, file_path, file_type=name_filter)
            except Exception as err:
                logger.exception(err)
                ExceptionDialog.show_exception('File Reader', err)

    def _copy_to_current_scan(self) -> None:
        item_index = self._get_current_item_index()

        if item_index >= 0:
            self._view.copier_dialog.destination_combo_box.setCurrentIndex(item_index)
            self._view.copier_dialog.open()

    def _finish_copying_scan(self, result: int) -> None:
        if result == QDialog.DialogCode.Accepted:
            source_index = self._view.copier_dialog.source_combo_box.currentIndex()
            destination_index = self._view.copier_dialog.destination_combo_box.currentIndex()
            self._api.copy_scan(source_index, destination_index)

    def _edit_current_scan(self) -> None:
        item_index = self._get_current_item_index()

        if item_index < 0:
            return

        item_name = self._repository.get_name(item_index)
        item = self._repository[item_index]
        dialog = self._editor_factory.create_editor_dialog(item_name, item, self._view)
        dialog.open()

    def _save_current_scan_to_file(self) -> None:
        item_index = self._get_current_item_index()

        if item_index < 0:
            return

        file_path, name_filter = self._file_dialog_factory.get_save_file_path(
            self._view,
            'Save Scan',
            name_filters=[nameFilter for nameFilter in self._api.get_save_file_filters()],
            selected_name_filter=self._api.get_save_file_filter(),
        )

        if file_path:
            try:
                self._api.save_scan(item_index, file_path, name_filter)
            except Exception as err:
                logger.exception(err)
                ExceptionDialog.show_exception('File Writer', err)

    def _sync_current_scan_to_settings(self) -> None:
        item_index = self._get_current_item_index()

        if item_index < 0:
            logger.warning('No current item!')
        else:
            item = self._repository[item_index]
            item.sync_to_settings()

    def _redraw_plot(self) -> None:
        self._plot_view.axes.clear()

        for row in range(self._table_proxy_model.rowCount()):
            proxy_index = self._table_proxy_model.index(row, 0)
            item_index = self._table_proxy_model.mapToSource(proxy_index).row()

            if self._table_model.is_item_checked(item_index):
                item_name = self._repository.get_name(item_index)
                scan = self._repository[item_index].get_scan()
                x = [point.position_x_m for point in scan]
                y = [point.position_y_m for point in scan]
                self._plot_view.axes.plot(x, y, '.-', label=item_name, linewidth=1.5)

        self._plot_view.axes.invert_yaxis()
        self._plot_view.axes.axis('equal')
        self._plot_view.axes.grid(True)
        self._plot_view.axes.set_xlabel('X [m]')
        self._plot_view.axes.set_ylabel('Y [m]')

        if len(self._plot_view.axes.lines) > 0:
            self._plot_view.axes.legend(loc='best')

        self._plot_view.figure_canvas.draw()

    def _estimate_transform(self) -> None:  # TODO
        item_index = self._get_current_item_index()

        if item_index < 0:
            logger.warning('No current item!')
            return

        _ = QMessageBox.information(
            self._view,
            'Not Implemented',
            'Affine transform estimator is not yet implemented.',
        )

    def _update_view(self, current: QModelIndex, previous: QModelIndex) -> None:
        enabled = current.isValid()
        self._view.button_box.load_button.setEnabled(enabled)
        self._view.button_box.save_button.setEnabled(enabled)
        self._view.button_box.edit_button.setEnabled(enabled)
        self._view.button_box.analyze_button.setEnabled(enabled)
        self._redraw_plot()

    def handle_item_inserted(self, index: int, item: ScanRepositoryItem) -> None:
        self._table_model.insert_item(index, item)

    def handle_item_changed(self, index: int, item: ScanRepositoryItem) -> None:
        self._table_model.update_item(index, item)

        if self._table_model.is_item_checked(index):
            self._redraw_plot()

    def handle_item_removed(self, index: int, item: ScanRepositoryItem) -> None:
        self._table_model.remove_item(index, item)

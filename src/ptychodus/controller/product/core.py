from __future__ import annotations
from collections.abc import Sequence
from typing import Any
import logging

from PyQt5.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QObject,
    QSortFilterProxyModel,
    Qt,
)
from PyQt5.QtGui import QBrush
from PyQt5.QtWidgets import QAbstractItemView, QAction

from ptychodus.api.product import LossValue
from ptychodus.api.units import BYTES_PER_MEGABYTE

from ...model.product import (
    ProductAPI,
    ProductRepository,
    ProductRepositoryItem,
    ProductRepositoryObserver,
)
from ...model.diffraction import AssembledDiffractionDataset
from ...model.product.metadata import MetadataRepositoryItem
from ...model.product.object import ObjectRepositoryItem
from ...model.product.probe import ProbeRepositoryItem
from ...model.product.scan import ScanRepositoryItem
from ...view.product import ProductView
from ...view.widgets import ExceptionDialog
from ..data import FileDialogFactory
from ..helpers import (
    connect_current_changed_signal,
    connect_triggered_signal,
    create_brush_for_editable_cell,
)
from .editor import ProductEditorViewController

logger = logging.getLogger(__name__)


class ProductRepositoryTableModel(QAbstractTableModel):
    def __init__(
        self,
        repository: ProductRepository,
        editable_item_brush: QBrush,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._repository = repository
        self._editable_item_brush = editable_item_brush
        self._header = [
            'Name',
            'Detector-Object\nDistance [m]',
            'Probe Energy\n[keV]',
            'Probe Photon\nCount',
            'Pixel Width\n[nm]',
            'Pixel Height\n[nm]',
            'Size\n[MB]',
        ]

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        value = super().flags(index)

        if index.isValid() and index.column() < 4:
            value |= Qt.ItemFlag.ItemIsEditable

        return value

    def headerData(  # noqa: N802
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return self._header[section]

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if index.isValid():
            try:
                item = self._repository[index.row()]
            except IndexError as err:
                logger.exception(err)
                return None

            metadata_item = item.get_metadata_item()
            geometry = item.get_geometry()

            if role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.EditRole:
                match index.column():
                    case 0:
                        return metadata_item.name.get_value()
                    case 1:
                        return f'{metadata_item.detector_distance_m.get_value():.4g}'
                    case 2:
                        return f'{metadata_item.probe_energy_eV.get_value() / 1e3:.4g}'
                    case 3:
                        return f'{metadata_item.probe_photon_count.get_value():.4g}'
                    case 4:
                        return f'{geometry.object_plane_pixel_width_m * 1e9:.4g}'
                    case 5:
                        return f'{geometry.object_plane_pixel_height_m * 1e9:.4g}'
                    case 6:
                        product = item.get_product()
                        return f'{product.nbytes / BYTES_PER_MEGABYTE:.2f}'
            elif role == Qt.ItemDataRole.BackgroundRole:
                if index.flags() & Qt.ItemFlag.ItemIsEditable:
                    return self._editable_item_brush

    def setData(self, index: QModelIndex, value: Any, role: int = Qt.ItemDataRole.EditRole) -> bool:  # noqa: N802
        if index.isValid() and role == Qt.ItemDataRole.EditRole:
            try:
                item = self._repository[index.row()]
            except IndexError as err:
                logger.exception(err)
                return False

            metadata_item = item.get_metadata_item()

            if index.column() == 0:
                metadata_item.name.set_value(str(value))
                return True
            elif index.column() == 1:
                try:
                    distance_m = float(value)
                except ValueError:
                    return False

                metadata_item.detector_distance_m.set_value(distance_m)
                return True
            elif index.column() == 2:
                try:
                    energy_keV = float(value)  # noqa: N806
                except ValueError:
                    return False

                metadata_item.probe_energy_eV.set_value(energy_keV * 1e3)
                return True
            elif index.column() == 3:
                try:
                    photon_count = float(value)
                except ValueError:
                    return False

                metadata_item.probe_photon_count.set_value(photon_count)
                return True

        return False

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self._repository)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self._header)


class ProductController(ProductRepositoryObserver):
    def __init__(
        self,
        dataset: AssembledDiffractionDataset,
        repository: ProductRepository,
        api: ProductAPI,
        view: ProductView,
        file_dialog_factory: FileDialogFactory,
        duplicate_action: QAction,
        table_model: ProductRepositoryTableModel,
        table_proxy_model: QSortFilterProxyModel,
    ) -> None:
        super().__init__()
        self._dataset = dataset
        self._repository = repository
        self._api = api
        self._view = view
        self._file_dialog_factory = file_dialog_factory
        self._duplicate_action = duplicate_action
        self._table_model = table_model
        self._table_proxy_model = table_proxy_model

    @classmethod
    def create_instance(
        cls,
        dataset: AssembledDiffractionDataset,
        repository: ProductRepository,
        api: ProductAPI,
        view: ProductView,
        file_dialog_factory: FileDialogFactory,
    ) -> ProductController:
        open_file_action = view.button_box.insert_menu.addAction('Open File...')
        create_new_action = view.button_box.insert_menu.addAction('Create New')
        duplicate_action = view.button_box.insert_menu.addAction('Duplicate')
        save_file_action = view.button_box.save_menu.addAction('Save File...')
        sync_to_settings_action = view.button_box.save_menu.addAction('Sync To Settings')

        editable_item_brush = create_brush_for_editable_cell(view.table_view)
        table_model = ProductRepositoryTableModel(repository, editable_item_brush)

        table_proxy_model = QSortFilterProxyModel()
        table_proxy_model.setSourceModel(table_model)

        controller = cls(
            dataset,
            repository,
            api,
            view,
            file_dialog_factory,
            duplicate_action,
            table_model,
            table_proxy_model,
        )
        repository.add_observer(controller)
        controller._update_info_text()

        view.table_view.setModel(table_proxy_model)
        view.table_view.setSortingEnabled(True)
        view.table_view.sortByColumn(0, Qt.SortOrder.AscendingOrder)
        vertical_header = view.table_view.verticalHeader()

        if vertical_header is not None:
            vertical_header.hide()

        view.table_view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        connect_current_changed_signal(view.table_view, controller._update_enabled_buttons)
        controller._update_enabled_buttons(QModelIndex(), QModelIndex())

        connect_triggered_signal(open_file_action, controller._open_product_from_file)
        connect_triggered_signal(create_new_action, controller._create_new_product)
        connect_triggered_signal(duplicate_action, controller._duplicate_current_product)
        connect_triggered_signal(save_file_action, controller._save_current_product_to_file)
        connect_triggered_signal(
            sync_to_settings_action, controller._sync_current_product_to_settings
        )

        view.button_box.edit_button.clicked.connect(controller._edit_current_product)
        view.button_box.remove_button.clicked.connect(controller._remove_current_product)

        return controller

    @property
    def table_model(self) -> QAbstractTableModel:
        return self._table_model

    def _get_current_item_index(self) -> int:
        proxy_index = self._view.table_view.currentIndex()

        if proxy_index.isValid():
            model_index = self._table_proxy_model.mapToSource(proxy_index)
            return model_index.row()

        logger.warning('No current index!')
        return -1

    def _open_product_from_file(self) -> None:
        file_path, name_filter = self._file_dialog_factory.get_open_file_path(
            self._view,
            'Open Product',
            name_filters=[nf for nf in self._api.get_open_file_filters()],
            selected_name_filter=self._api.get_open_file_filter(),
        )

        if file_path:
            try:
                self._api.open_product(file_path, file_type=name_filter)
            except Exception as err:
                logger.exception(err)
                ExceptionDialog.show_exception('File Reader', err)

    def _create_new_product(self) -> None:
        self._api.insert_new_product()

    def _save_current_product_to_file(self) -> None:
        current = self._table_proxy_model.mapToSource(self._view.table_view.currentIndex())

        if current.isValid():
            file_path, name_filter = self._file_dialog_factory.get_save_file_path(
                self._view,
                'Save Product',
                name_filters=[nf for nf in self._api.get_save_file_filters()],
                selected_name_filter=self._api.get_save_file_filter(),
            )

            if file_path:
                try:
                    self._api.save_product(current.row(), file_path, file_type=name_filter)
                except Exception as err:
                    logger.exception(err)
                    ExceptionDialog.show_exception('File Writer', err)
        else:
            logger.error('No current item!')

    def _sync_current_product_to_settings(self) -> None:
        item_index = self._get_current_item_index()

        if item_index < 0:
            logger.warning('No current item!')
        else:
            item = self._repository[item_index]
            item.sync_to_settings()

    def _duplicate_current_product(self) -> None:
        current = self._table_proxy_model.mapToSource(self._view.table_view.currentIndex())

        if current.isValid():
            like_item = self._repository[current.row()]
            self._api.insert_product(like_item.get_product())
        else:
            logger.error('No current item!')

    def _edit_current_product(self) -> None:
        current = self._table_proxy_model.mapToSource(self._view.table_view.currentIndex())

        if current.isValid():
            product = self._repository[current.row()]
            ProductEditorViewController.edit_product(self._dataset, product, self._view)
        else:
            logger.error('No current item!')

    def _remove_current_product(self) -> None:
        current = self._table_proxy_model.mapToSource(self._view.table_view.currentIndex())

        if current.isValid():
            self._repository.remove_product(current.row())
        else:
            logger.error('No current item!')

    def _update_enabled_buttons(self, current: QModelIndex, previous: QModelIndex) -> None:
        enabled = current.isValid()
        self._duplicate_action.setEnabled(enabled)
        self._view.button_box.save_button.setEnabled(enabled)
        self._view.button_box.edit_button.setEnabled(enabled)
        self._view.button_box.remove_button.setEnabled(enabled)

    def _update_info_text(self) -> None:
        info_text = self._repository.get_info_text()
        self._view.info_label.setText(info_text)

    def handle_item_inserted(self, index: int, item: ProductRepositoryItem) -> None:
        parent = QModelIndex()
        self._table_model.beginInsertRows(parent, index, index)
        self._table_model.endInsertRows()
        self._update_info_text()

    def handle_metadata_changed(self, index: int, item: MetadataRepositoryItem) -> None:
        top_left = self._table_model.index(index, 0)
        bottom_right = self._table_model.index(index, self._table_model.columnCount() - 1)
        self._table_model.dataChanged.emit(top_left, bottom_right, [Qt.ItemDataRole.DisplayRole])
        self._update_info_text()

    def handle_scan_changed(self, index: int, item: ScanRepositoryItem) -> None:
        self._update_info_text()

    def handle_probe_changed(self, index: int, item: ProbeRepositoryItem) -> None:
        self._update_info_text()

    def handle_object_changed(self, index: int, item: ObjectRepositoryItem) -> None:
        self._update_info_text()

    def handle_losses_changed(self, index: int, losses: Sequence[LossValue]) -> None:
        self._update_info_text()

    def handle_item_removed(self, index: int, item: ProductRepositoryItem) -> None:
        parent = QModelIndex()
        self._table_model.beginRemoveRows(parent, index, index)
        self._table_model.endRemoveRows()
        self._update_info_text()

from __future__ import annotations
from collections.abc import Sequence
from enum import IntEnum
from typing import Any, cast
import logging

from PyQt5.QtCore import (
    QAbstractTableModel,
    QIdentityProxyModel,
    QModelIndex,
    QObject,
    QSortFilterProxyModel,
    Qt,
)
from PyQt5.QtGui import QBrush, QFont
from PyQt5.QtWidgets import QAbstractItemView, QAction, QInputDialog

from ptychodus.api.constants import (
    ONE_KILOELECTRONVOLT_EV,
    ONE_NANOMETER_M,
    format_bytes,
)
from ptychodus.api.product import LossValue

from ...model.diffraction import AssembledDiffractionDataset, DiffractionDatasetRepository
from ..diffraction.dataset import UNBOUND_DATASET, DiffractionDatasetComboModel
from ...model.product import (
    ProductAPI,
    ProductRepository,
    ProductRepositoryItem,
    ProductRepositoryObserver,
)
from ...model.product.metadata import MetadataRepositoryItem
from ...model.product.object import ObjectRepositoryItem
from ...model.product.probe import ProbeRepositoryItem
from ...model.product.probe_positions import ProbePositionsRepositoryItem
from ...view.product import ProductView
from ...view.widgets import ComboBoxItemDelegate, ExceptionDialog
from ..data import FileDialogFactory
from ..helpers import UserRoleSortFilterProxyModel, connect_triggered_signal
from .editor import ProductEditorViewController

logger = logging.getLogger(__name__)


class _Column(IntEnum):
    """Column order of ProductRepositoryTableModel.

    An enum rather than module-level ints so that ``match`` arms are value patterns:
    a bare name would be a capture pattern that swallows every column.
    """

    NAME = 0
    DIFFRACTION_DATASET = 1
    DETECTOR_DISTANCE_M = 2
    PROBE_ENERGY_KEV = 3
    PROBE_PHOTON_COUNT = 4
    PIXEL_WIDTH_NM = 5
    PIXEL_HEIGHT_NM = 6
    SIZE = 7


_EDITABLE_COLUMNS = frozenset(
    {
        _Column.NAME,
        _Column.DIFFRACTION_DATASET,
        _Column.DETECTOR_DISTANCE_M,
        _Column.PROBE_ENERGY_KEV,
        _Column.PROBE_PHOTON_COUNT,
    }
)


class ProductRepositoryTableModel(QAbstractTableModel):
    """Table model over ProductRepository.

    Registers itself as a ProductRepositoryObserver and translates repository
    change callbacks into Qt structural / dataChanged signals. Duck-typed
    against ProductRepositoryObserver — inheriting the ABC would clash with
    sip's wrappertype metaclass on QAbstractTableModel.
    """

    def __init__(
        self,
        repository: ProductRepository,
        diffraction_repository: DiffractionDatasetRepository,
        editable_item_brush: QBrush,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._repository = repository
        self._diffraction_repository = diffraction_repository
        self._editable_item_brush = editable_item_brush
        self._header = [
            'Name',
            'Diffraction\nDataset',
            'Detector-Object\nDistance [m]',
            'Probe Energy\n[keV]',
            'Probe Photon\nCount',
            'Pixel Width\n[nm]',
            'Pixel Height\n[nm]',
            'Size',
        ]
        # Duck-typed; see class docstring on the ABC / sip metaclass conflict.
        repository.add_observer(cast(ProductRepositoryObserver, self))

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        value = super().flags(index)

        if index.isValid() and index.column() in _EDITABLE_COLUMNS:
            try:
                item = self._repository[index.row()]
            except IndexError:
                return value

            if not item.is_pending() and not item.is_failed():
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
            pending = item.is_pending()
            failed = item.is_failed()

            if role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.EditRole:
                match index.column():
                    case _Column.NAME:
                        return metadata_item.name.get_value()
                    case _Column.DIFFRACTION_DATASET:
                        if pending or failed:
                            return '—'
                        dataset = item.get_dataset()
                        return UNBOUND_DATASET if dataset is None else dataset.get_name()
                    case _Column.DETECTOR_DISTANCE_M:
                        if pending or failed:
                            return '—'
                        return f'{metadata_item.detector_distance_m.get_value():.4g}'
                    case _Column.PROBE_ENERGY_KEV:
                        if pending or failed:
                            return '—'
                        return f'{metadata_item.probe_energy_eV.get_value() / ONE_KILOELECTRONVOLT_EV:.4g}'
                    case _Column.PROBE_PHOTON_COUNT:
                        if pending or failed:
                            return '—'
                        return f'{metadata_item.probe_photon_count.get_value():.4g}'
                    case _Column.PIXEL_WIDTH_NM:
                        if pending or failed:
                            return '—'
                        return f'{geometry.get_object_plane_pixel_geometry().width_m / ONE_NANOMETER_M:.4g}'
                    case _Column.PIXEL_HEIGHT_NM:
                        if pending or failed:
                            return '—'
                        return f'{geometry.get_object_plane_pixel_geometry().height_m / ONE_NANOMETER_M:.4g}'
                    case _Column.SIZE:
                        if pending or failed:
                            return '—'
                        return format_bytes(item.get_product().nbytes)
            elif role == Qt.ItemDataRole.UserRole:
                if index.column() == _Column.SIZE:
                    # Numeric sort key: the display text mixes units and cannot be
                    # ordered as a string. Pending and failed rows sort as zero so the
                    # comparator stays total.
                    return 0 if pending or failed else item.get_product().nbytes
            elif role == Qt.ItemDataRole.FontRole:
                if pending or failed:
                    font = QFont()
                    font.setItalic(pending)
                    font.setStrikeOut(failed)
                    return font
            elif role == Qt.ItemDataRole.ToolTipRole:
                if pending:
                    return 'Loading…'
                if failed:
                    return 'Load failed'
                if index.column() == _Column.DIFFRACTION_DATASET and item.get_dataset() is None:
                    # Surface why Reconstruct / Train are greyed out in the Processing
                    # panel; otherwise an unbound product fails silently.
                    return 'No diffraction dataset — reconstruction unavailable'
            elif role == Qt.ItemDataRole.ForegroundRole:
                if pending or failed:
                    return QBrush(Qt.GlobalColor.gray)
                if index.column() == _Column.DIFFRACTION_DATASET and item.get_dataset() is None:
                    return QBrush(Qt.GlobalColor.gray)
            elif role == Qt.ItemDataRole.BackgroundRole:
                if not (pending or failed) and (index.flags() & Qt.ItemFlag.ItemIsEditable):
                    return self._editable_item_brush

    def setData(self, index: QModelIndex, value: Any, role: int = Qt.ItemDataRole.EditRole) -> bool:  # noqa: N802
        if index.isValid() and role == Qt.ItemDataRole.EditRole:
            try:
                item = self._repository[index.row()]
            except IndexError as err:
                logger.exception(err)
                return False

            metadata_item = item.get_metadata_item()

            if index.column() == _Column.NAME:
                metadata_item.name.set_value(str(value))
                return True
            elif index.column() == _Column.DIFFRACTION_DATASET:
                text = str(value)

                if text == UNBOUND_DATASET:
                    item.unbind_dataset()
                    return True

                for dataset in self._diffraction_repository:
                    if dataset.get_name() == text:
                        item.bind_dataset(dataset)
                        return True

                return False
            elif index.column() == _Column.DETECTOR_DISTANCE_M:
                try:
                    distance_m = float(value)
                except ValueError:
                    return False

                metadata_item.detector_distance_m.set_value(distance_m)
                return True
            elif index.column() == _Column.PROBE_ENERGY_KEV:
                try:
                    energy_keV = float(value)  # noqa: N806
                except ValueError:
                    return False

                metadata_item.probe_energy_eV.set_value(energy_keV * ONE_KILOELECTRONVOLT_EV)
                return True
            elif index.column() == _Column.PROBE_PHOTON_COUNT:
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

    def _emit_row_changed(self, index: int, roles: list[int]) -> None:
        top_left = self.index(index, 0)
        bottom_right = self.index(index, self.columnCount() - 1)
        self.dataChanged.emit(top_left, bottom_right, roles)

    def handle_item_inserted(self, index: int, item: ProductRepositoryItem) -> None:
        self.beginInsertRows(QModelIndex(), index, index)
        self.endInsertRows()

    def handle_item_removed(self, index: int, item: ProductRepositoryItem) -> None:
        self.beginRemoveRows(QModelIndex(), index, index)
        self.endRemoveRows()

    def handle_metadata_changed(self, index: int, item: MetadataRepositoryItem) -> None:
        self._emit_row_changed(index, [Qt.ItemDataRole.DisplayRole])

    def handle_state_changed(self, index: int, item: ProductRepositoryItem) -> None:
        self._emit_row_changed(
            index,
            [
                Qt.ItemDataRole.DisplayRole,
                Qt.ItemDataRole.ForegroundRole,
                Qt.ItemDataRole.FontRole,
                Qt.ItemDataRole.ToolTipRole,
            ],
        )

    def handle_probe_positions_changed(
        self, index: int, item: ProbePositionsRepositoryItem
    ) -> None:
        pass

    def handle_probe_changed(self, index: int, item: ProbeRepositoryItem) -> None:
        pass

    def handle_object_changed(self, index: int, item: ObjectRepositoryItem) -> None:
        pass

    def handle_losses_changed(self, index: int, losses: Sequence[LossValue]) -> None:
        pass

    def handle_dataset_changed(self, index: int, item: ProductRepositoryItem) -> None:
        # Repaints the whole row: rebinding changes the Diffraction Dataset cell and
        # re-syncs the geometry-derived pixel columns. Also the only signal that a
        # dataset removal silently unbound this product.
        self._emit_row_changed(
            index,
            [
                Qt.ItemDataRole.DisplayRole,
                Qt.ItemDataRole.ToolTipRole,
                Qt.ItemDataRole.ForegroundRole,
            ],
        )


class ProductRepositoryComboProxyModel(QIdentityProxyModel):
    """Presents ProductRepositoryTableModel as a single-column, non-editable
    list suitable for a QComboBox: strips ItemIsEditable and disables pending /
    failed items so they cannot be selected.
    """

    def __init__(
        self,
        source_model: ProductRepositoryTableModel,
        repository: ProductRepository,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._repository = repository
        self.setSourceModel(source_model)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 1

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        base = super().flags(index)

        if not index.isValid():
            return base

        base &= ~Qt.ItemFlag.ItemIsEditable

        try:
            item = self._repository[index.row()]
        except IndexError:
            return base

        if item.is_pending() or item.is_failed():
            return Qt.ItemFlags(Qt.NoItemFlags)

        return base


class ProductController(ProductRepositoryObserver):
    def __init__(
        self,
        repository: ProductRepository,
        api: ProductAPI,
        diffraction_repository: DiffractionDatasetRepository,
        view: ProductView,
        file_dialog_factory: FileDialogFactory,
        duplicate_action: QAction,
        table_model: ProductRepositoryTableModel,
        table_proxy_model: QSortFilterProxyModel,
    ) -> None:
        super().__init__()
        self._repository = repository
        self._api = api
        self._diffraction_repository = diffraction_repository
        self._view = view
        self._file_dialog_factory = file_dialog_factory
        self._duplicate_action = duplicate_action
        self._table_model = table_model
        self._table_proxy_model = table_proxy_model
        self._dataset_combo_model = DiffractionDatasetComboModel(
            diffraction_repository, unbound_label=UNBOUND_DATASET
        )

    @classmethod
    def create_instance(
        cls,
        repository: ProductRepository,
        api: ProductAPI,
        diffraction_repository: DiffractionDatasetRepository,
        table_model: ProductRepositoryTableModel,
        view: ProductView,
        file_dialog_factory: FileDialogFactory,
    ) -> ProductController:
        open_file_action = view.button_box.insert_menu.addAction('Open File...')
        create_new_action = view.button_box.insert_menu.addAction('Create New')
        duplicate_action = view.button_box.insert_menu.addAction('Duplicate')
        save_file_action = view.button_box.save_menu.addAction('Save File...')
        sync_to_settings_action = view.button_box.save_menu.addAction('Sync To Settings')

        table_proxy_model = UserRoleSortFilterProxyModel()
        table_proxy_model.setSourceModel(table_model)

        controller = cls(
            repository,
            api,
            diffraction_repository,
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
        view.table_view.setItemDelegateForColumn(
            _Column.DIFFRACTION_DATASET,
            ComboBoxItemDelegate(controller._dataset_combo_model, view.table_view),
        )
        header = view.table_view.horizontalHeader()
        header.setSectionResizeMode(header.ResizeMode.ResizeToContents)
        header.setHighlightSections(False)

        selection_model = view.table_view.selectionModel()
        if selection_model is not None:
            selection_model.currentChanged.connect(controller._update_enabled_buttons)
        controller._update_enabled_buttons(QModelIndex(), QModelIndex())

        # Auto-select first row on insert if nothing is currently selected;
        # re-select a neighbor on removal if the current row went away.
        table_proxy_model.rowsInserted.connect(controller._on_rows_inserted)
        table_proxy_model.rowsRemoved.connect(controller._on_rows_removed)
        controller._auto_select_first_row_if_empty()

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

    def get_current_item_index(self) -> int:
        """Repository row index of the currently-selected product, or -1 if none is selected."""
        proxy_index = self._view.table_view.currentIndex()

        if proxy_index.isValid():
            model_index = self._table_proxy_model.mapToSource(proxy_index)
            return model_index.row()

        return -1

    def _choose_dataset(self, title: str) -> tuple[bool, AssembledDiffractionDataset | None]:
        """Prompt the user to bind the new product to a diffraction dataset.

        Returns (accepted, dataset). ``dataset`` is None when the user picks the
        unbound entry. Shares the combo model with the table column and the editor
        dialog so all three offer the same list.
        """
        model = self._dataset_combo_model
        labels = [str(model.index(row, 0).data()) for row in range(model.rowCount())]

        label, accepted = QInputDialog.getItem(
            self._view,
            title,
            'Diffraction Dataset:',
            labels,
            0,
            False,
        )

        if not accepted:
            return False, None

        return True, model.dataset_at(labels.index(label))

    def _open_product_from_file(self) -> None:
        file_path, name_filter = self._file_dialog_factory.get_open_file_path(
            self._view,
            'Open Product',
            name_filters=[nf for nf in self._api.get_open_file_filters()],
            selected_name_filter=self._api.get_open_file_filter(),
        )

        if file_path:
            accepted, dataset = self._choose_dataset('Open Product')

            if not accepted:
                return

            try:
                self._api.open_product(
                    file_path, file_type=name_filter, dataset=dataset, block=False
                )
            except Exception as err:
                logger.exception(err)
                ExceptionDialog.show_exception('File Reader', err)

    def _create_new_product(self) -> None:
        accepted, dataset = self._choose_dataset('Create Product')

        if not accepted:
            return

        self._api.insert_new_product(dataset=dataset, block=False)

    def _save_current_product_to_file(self) -> None:
        item_index = self.get_current_item_index()

        if item_index >= 0:
            file_path, name_filter = self._file_dialog_factory.get_save_file_path(
                self._view,
                'Save Product',
                name_filters=[nf for nf in self._api.get_save_file_filters()],
                selected_name_filter=self._api.get_save_file_filter(),
            )

            if file_path:
                try:
                    self._api.save_product(item_index, file_path, file_type=name_filter)
                except Exception as err:
                    logger.exception(err)
                    ExceptionDialog.show_exception('File Writer', err)
        else:
            logger.error('No current item!')

    def _sync_current_product_to_settings(self) -> None:
        item_index = self.get_current_item_index()

        if item_index < 0:
            logger.warning('No current item!')
        else:
            item = self._repository[item_index]
            item.sync_to_settings()

    def _duplicate_current_product(self) -> None:
        item_index = self.get_current_item_index()

        if item_index >= 0:
            like_item = self._repository[item_index]
            self._api.insert_product(
                like_item.get_product(), dataset=like_item.get_dataset(), block=False
            )
        else:
            logger.error('No current item!')

    def _edit_current_product(self) -> None:
        item_index = self.get_current_item_index()

        if item_index >= 0:
            product = self._repository[item_index]
            ProductEditorViewController.edit_product(
                self._diffraction_repository, self._dataset_combo_model, product, self._view
            )
        else:
            logger.error('No current item!')

    def _remove_current_product(self) -> None:
        item_index = self.get_current_item_index()

        if item_index >= 0:
            self._repository.remove_product(item_index)
        else:
            logger.error('No current item!')

    def _update_enabled_buttons(self, current: QModelIndex, previous: QModelIndex) -> None:
        source_index = (
            self._table_proxy_model.mapToSource(current) if current.isValid() else current
        )
        enabled = source_index.isValid()

        ready = False
        if enabled:
            try:
                item = self._repository[source_index.row()]
            except IndexError:
                ready = False
            else:
                ready = not item.is_pending()

        self._duplicate_action.setEnabled(ready)
        self._view.button_box.save_button.setEnabled(ready)
        self._view.button_box.edit_button.setEnabled(ready)
        # Remove is always safe: it drops the row whether pending, failed, or ready.
        self._view.button_box.remove_button.setEnabled(enabled)

    def _update_info_text(self) -> None:
        info_text = self._repository.get_info_text()
        self._view.info_label.setText(info_text)

    def _auto_select_first_row_if_empty(self) -> None:
        if self._view.table_view.currentIndex().isValid():
            return

        if self._table_proxy_model.rowCount() > 0:
            self._view.table_view.setCurrentIndex(self._table_proxy_model.index(0, 0))

    def _on_rows_inserted(self, parent: QModelIndex, first: int, last: int) -> None:
        # Auto-select the newly-inserted row if nothing is currently selected.
        self._auto_select_first_row_if_empty()

    def _on_rows_removed(self, parent: QModelIndex, first: int, last: int) -> None:
        # Qt clears the current index when the current row is removed; pick a
        # neighbor at the same proxy position so the user is never left without
        # a selection while rows remain.
        if self._view.table_view.currentIndex().isValid():
            return

        row_count = self._table_proxy_model.rowCount()
        if row_count > 0:
            target = min(first, row_count - 1)
            self._view.table_view.setCurrentIndex(self._table_proxy_model.index(target, 0))

    def handle_item_inserted(self, index: int, item: ProductRepositoryItem) -> None:
        self._update_info_text()

    def handle_metadata_changed(self, index: int, item: MetadataRepositoryItem) -> None:
        self._update_info_text()

    def handle_probe_positions_changed(
        self, index: int, item: ProbePositionsRepositoryItem
    ) -> None:
        self._update_info_text()

    def handle_probe_changed(self, index: int, item: ProbeRepositoryItem) -> None:
        self._update_info_text()

    def handle_object_changed(self, index: int, item: ObjectRepositoryItem) -> None:
        self._update_info_text()

    def handle_losses_changed(self, index: int, losses: Sequence[LossValue]) -> None:
        self._update_info_text()

    def handle_dataset_changed(self, index: int, item: ProductRepositoryItem) -> None:
        self._update_info_text()

    def handle_state_changed(self, index: int, item: ProductRepositoryItem) -> None:
        self._update_info_text()

        if self.get_current_item_index() == index:
            current = self._view.table_view.currentIndex()
            self._update_enabled_buttons(current, QModelIndex())

    def handle_item_removed(self, index: int, item: ProductRepositoryItem) -> None:
        self._update_info_text()

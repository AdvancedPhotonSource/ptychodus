from __future__ import annotations
from collections.abc import Sequence
from enum import Enum
from typing import Any
import logging

from PyQt5.QtCore import Qt, QAbstractTableModel, QModelIndex, QObject, QStringListModel
from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QTableView

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry

from ..model.product import ProductRepository
from ..view.settings import SettingsView, SyncProductToSettingsDialog
from .data import FileDialogFactory
from .product.core import ProductRepositoryComboProxyModel, ProductRepositoryTableModel

logger = logging.getLogger(__name__)


class _Column(Enum):
    """Every column of SettingsTableModel: its order and its header.

    The member value is the header and the position comes from declaration order,
    so a column cannot be added to the ordering and forgotten in the headers.
    Bare names in a ``match`` arm are capture patterns, not value patterns.
    """

    NAME = 'Name'
    VALUE = 'Value'

    def __init__(self, header: str) -> None:
        self.header = header
        # Members already defined when this one is constructed, which is its position.
        self.index = len(self.__class__.__members__)

    @classmethod
    def from_index(cls, index: int) -> _Column:
        """The column at `index`.

        Raises:
            IndexError: for a column outside the table, which the model's own
                ``columnCount`` makes impossible and so would be a bug rather than input.
        """
        if index < 0:
            raise IndexError(f'Column index {index} is negative.')

        return list(cls)[index]


class SettingsTableModel(QAbstractTableModel):
    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._names: Sequence[str] = list()
        self._values: Sequence[str] = list()

    def set_names_and_values(self, names: Sequence[str], values: Sequence[str]) -> None:
        self.beginResetModel()
        self._names = names
        self._values = values
        self.endResetModel()

    def headerData(  # noqa: N802
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return _Column.from_index(section).header

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if index.isValid() and role == Qt.ItemDataRole.DisplayRole:
            column = _Column.from_index(index.column())

            if column is _Column.NAME:
                return self._names[index.row()]
            elif column is _Column.VALUE:
                return str(self._values[index.row()])

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self._names)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(_Column)


class SettingsController(Observer):
    def __init__(
        self,
        settings_registry: SettingsRegistry,
        product_repository: ProductRepository,
        product_table_model: ProductRepositoryTableModel,
        settings_view: SettingsView,
        settings_table_view: QTableView,
        file_dialog_factory: FileDialogFactory,
    ) -> None:
        super().__init__()
        self._settings_registry = settings_registry
        self._product_repository = product_repository
        self._settings_view = settings_view
        self._settings_table_view = settings_table_view
        self._file_dialog_factory = file_dialog_factory

        self._settings_list_model = QStringListModel()
        self._settings_table_model = SettingsTableModel()
        self._product_combo_model = ProductRepositoryComboProxyModel(
            product_table_model, product_repository
        )
        self._sync_dialog = SyncProductToSettingsDialog(settings_view)

        settings_registry.add_observer(self)

        settings_view.list_view.setModel(self._settings_list_model)
        settings_selection_model = settings_view.list_view.selectionModel()

        if settings_selection_model is None:
            raise ValueError('selection_model is None!')
        else:
            settings_selection_model.currentChanged.connect(self._update_view)

        self._settings_table_view.setModel(self._settings_table_model)
        header = self._settings_table_view.horizontalHeader()
        header.setSectionResizeMode(header.ResizeMode.ResizeToContents)

        open_button = settings_view.button_box.button(QDialogButtonBox.StandardButton.Open)
        open_button.clicked.connect(self._open_settings)
        save_button = settings_view.button_box.button(QDialogButtonBox.StandardButton.Save)
        save_button.clicked.connect(self._sync_dialog.open)

        self._sync_dialog.product_combo_box.setModel(self._product_combo_model)
        self._sync_dialog.finished.connect(self._save_settings)

        self._sync_model_to_view()

    def _open_settings(self) -> None:
        file_path, _ = self._file_dialog_factory.get_open_file_path(
            self._settings_view,
            'Open Settings',
            name_filters=self._settings_registry.get_open_file_filters(),
            selected_name_filter=self._settings_registry.get_open_file_filter(),
        )

        if file_path:
            self._settings_registry.open_settings(file_path)

    def _save_settings(self, result: int) -> None:
        if result == QDialog.DialogCode.Accepted:
            item_index = self._sync_dialog.product_combo_box.currentIndex()

            if item_index < 0:
                logger.warning('No current item!')
            else:
                item = self._product_repository[item_index]
                item.sync_to_settings()

        file_path, _ = self._file_dialog_factory.get_save_file_path(
            self._settings_view,
            'Save Settings',
            name_filters=self._settings_registry.get_save_file_filters(),
            selected_name_filter=self._settings_registry.get_save_file_filter(),
        )

        if file_path:
            self._settings_registry.save_settings(file_path)

    def _update_view(self, current: QModelIndex, previous: QModelIndex) -> None:
        if not current.isValid():
            return

        group_name = self._settings_list_model.data(current, Qt.ItemDataRole.DisplayRole)
        group = self._settings_registry[group_name]
        names: list[str] = list()
        values: list[str] = list()

        for parameter_name, parameter in group.parameters().items():
            names.append(parameter_name)
            values.append(parameter.get_value_as_string())

        self._settings_table_model.set_names_and_values(names, values)

    def _sync_model_to_view(self) -> None:
        self._settings_list_model.setStringList(sorted(iter(self._settings_registry)))

        current = self._settings_view.list_view.currentIndex()
        self._update_view(current, QModelIndex())

    def _update(self, observable: Observable) -> None:
        if observable is self._settings_registry:
            self._sync_model_to_view()

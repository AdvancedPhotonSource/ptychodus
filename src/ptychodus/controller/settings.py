from __future__ import annotations
from collections.abc import Sequence
from typing import Any

from PyQt5.QtCore import Qt, QAbstractTableModel, QModelIndex, QObject, QStringListModel
from PyQt5.QtWidgets import QTableView

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry

from ..view.settings import SettingsView
from .data import FileDialogFactory


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
            if section == 0:
                return 'Name'
            elif section == 1:
                return 'Value'

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if index.isValid() and role == Qt.ItemDataRole.DisplayRole:
            if index.column() == 0:
                return self._names[index.row()]
            elif index.column() == 1:
                return str(self._values[index.row()])

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self._names)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return 2


class SettingsController(Observer):
    def __init__(
        self,
        settings_registry: SettingsRegistry,
        view: SettingsView,
        table_view: QTableView,
        file_dialog_factory: FileDialogFactory,
    ) -> None:
        super().__init__()
        self._settings_registry = settings_registry
        self._view = view
        self._table_view = table_view
        self._file_dialog_factory = file_dialog_factory

        self._list_model = QStringListModel()
        self._table_model = SettingsTableModel()

        settings_registry.add_observer(self)

        view.list_view.setModel(self._list_model)
        selection_model = view.list_view.selectionModel()

        if selection_model is None:
            raise ValueError('selection_model is None!')
        else:
            selection_model.currentChanged.connect(self._update_view)

        self._table_view.setModel(self._table_model)

        view.button_box.open_button.clicked.connect(self._open_settings)
        view.button_box.save_button.clicked.connect(self._save_settings)

        self._sync_model_to_view()

    def _open_settings(self) -> None:
        file_path, _ = self._file_dialog_factory.get_open_file_path(
            self._view,
            'Open Settings',
            name_filters=self._settings_registry.get_open_file_filters(),
            selected_name_filter=self._settings_registry.get_open_file_filter(),
        )

        if file_path:
            self._settings_registry.open_settings(file_path)

    def _save_settings(self) -> None:
        file_path, _ = self._file_dialog_factory.get_save_file_path(
            self._view,
            'Save Settings',
            name_filters=self._settings_registry.get_save_file_filters(),
            selected_name_filter=self._settings_registry.get_save_file_filter(),
        )

        if file_path:
            self._settings_registry.save_settings(file_path)

    def _update_view(self, current: QModelIndex, previous: QModelIndex) -> None:
        if not current.isValid():
            return

        group_name = self._list_model.data(current, Qt.ItemDataRole.DisplayRole)
        group = self._settings_registry[group_name]
        names: list[str] = list()
        values: list[str] = list()

        for parameter_name, parameter in group.parameters().items():
            names.append(parameter_name)
            values.append(parameter.get_value_as_string())

        self._table_model.set_names_and_values(names, values)

    def _sync_model_to_view(self) -> None:
        self._list_model.setStringList(sorted(iter(self._settings_registry)))

        current = self._view.list_view.currentIndex()
        self._update_view(current, QModelIndex())

    def _update(self, observable: Observable) -> None:
        if observable is self._settings_registry:
            self._sync_model_to_view()

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

    def setNamesAndValues(self, names: Sequence[str], values: Sequence[str]) -> None:
        self.beginResetModel()
        self._names = names
        self._values = values
        self.endResetModel()

    def headerData(
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

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._names)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 2


class SettingsController(Observer):
    def __init__(
        self,
        settingsRegistry: SettingsRegistry,
        view: SettingsView,
        tableView: QTableView,
        fileDialogFactory: FileDialogFactory,
    ) -> None:
        super().__init__()
        self._settingsRegistry = settingsRegistry
        self._view = view
        self._tableView = tableView
        self._fileDialogFactory = fileDialogFactory

        self._listModel = QStringListModel()
        self._tableModel = SettingsTableModel()

        settingsRegistry.add_observer(self)

        view.listView.setModel(self._listModel)
        view.listView.selectionModel().currentChanged.connect(self._updateView)

        self._tableView.setModel(self._tableModel)

        view.buttonBox.openButton.clicked.connect(self._openSettings)
        view.buttonBox.saveButton.clicked.connect(self._saveSettings)

        self._syncModelToView()

    def _openSettings(self) -> None:
        filePath, _ = self._fileDialogFactory.getOpenFilePath(
            self._view,
            'Open Settings',
            nameFilters=self._settingsRegistry.get_open_file_filters(),
            selectedNameFilter=self._settingsRegistry.get_open_file_filter(),
        )

        if filePath:
            self._settingsRegistry.open_settings(filePath)

    def _saveSettings(self) -> None:
        filePath, _ = self._fileDialogFactory.get_save_file_path(
            self._view,
            'Save Settings',
            name_filters=self._settingsRegistry.get_save_file_filters(),
            selected_name_filter=self._settingsRegistry.get_save_file_filter(),
        )

        if filePath:
            self._settingsRegistry.save_settings(filePath)

    def _updateView(self, current: QModelIndex, previous: QModelIndex) -> None:
        if not current.isValid():
            return

        groupName = self._listModel.data(current, Qt.DisplayRole)
        group = self._settingsRegistry[groupName]
        names: list[str] = list()
        values: list[str] = list()

        for parameterName, parameter in group.parameters().items():
            names.append(parameterName)
            values.append(parameter.get_value_as_string())

        self._tableModel.setNamesAndValues(names, values)

    def _syncModelToView(self) -> None:
        self._listModel.setStringList(sorted(iter(self._settingsRegistry)))

        current = self._view.listView.currentIndex()
        self._updateView(current, QModelIndex())

    def _update(self, observable: Observable) -> None:
        if observable is self._settingsRegistry:
            self._syncModelToView()

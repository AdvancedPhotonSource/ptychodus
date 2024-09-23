from __future__ import annotations
from typing import Any

from PyQt5.QtCore import Qt, QAbstractListModel, QAbstractTableModel, QModelIndex, QObject
from PyQt5.QtWidgets import QTableView

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsGroup, SettingsRegistry

from ..view.settings import SettingsView
from .data import FileDialogFactory


class SettingsListModel(QAbstractListModel):

    def __init__(self, settingsRegistry: SettingsRegistry, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._settingsRegistry = settingsRegistry

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if index.isValid() and role == Qt.ItemDataRole.DisplayRole:
            settingsGroup = self._settingsRegistry[index.row()]
            return settingsGroup.name

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._settingsRegistry)


class SettingsTableModel(QAbstractTableModel):

    def __init__(self, settingsGroup: SettingsGroup | None, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._settingsGroup = settingsGroup

    def headerData(self,
                   section: int,
                   orientation: Qt.Orientation,
                   role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            if section == 0:
                return 'Name'
            elif section == 1:
                return 'Value'

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if self._settingsGroup is None:
            return None

        if index.isValid() and role == Qt.ItemDataRole.DisplayRole:
            nameList = list(self._settingsGroup.keys())
            name = nameList[index.row()]

            if index.column() == 0:
                return name
            elif index.column() == 1:
                return str(self._settingsGroup[name])

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._settingsGroup) if self._settingsGroup else 0

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 2


class SettingsController(Observer):

    def __init__(self, settingsRegistry: SettingsRegistry, view: SettingsView,
                 tableView: QTableView, fileDialogFactory: FileDialogFactory) -> None:
        super().__init__()
        self._settingsRegistry = settingsRegistry
        self._listModel = SettingsListModel(settingsRegistry)
        self._view = view
        self._tableView = tableView
        self._fileDialogFactory = fileDialogFactory

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry, view: SettingsView,
                       tableView: QTableView,
                       fileDialogFactory: FileDialogFactory) -> SettingsController:
        controller = cls(settingsRegistry, view, tableView, fileDialogFactory)
        settingsRegistry.addObserver(controller)

        view.listView.setModel(controller._listModel)
        view.listView.selectionModel().currentChanged.connect(controller._updateView)

        view.buttonBox.openButton.clicked.connect(controller._openSettings)
        view.buttonBox.saveButton.clicked.connect(controller._saveSettings)

        controller._syncModelToView()

        return controller

    def _openSettings(self) -> None:
        filePath, _ = self._fileDialogFactory.getOpenFilePath(
            self._view,
            'Open Settings',
            nameFilters=self._settingsRegistry.getOpenFileFilterList(),
            selectedNameFilter=self._settingsRegistry.getOpenFileFilter())

        if filePath:
            self._settingsRegistry.openSettings(filePath)

    def _saveSettings(self) -> None:
        filePath, _ = self._fileDialogFactory.getSaveFilePath(
            self._view,
            'Save Settings',
            nameFilters=self._settingsRegistry.getSaveFileFilterList(),
            selectedNameFilter=self._settingsRegistry.getSaveFileFilter())

        if filePath:
            self._settingsRegistry.saveSettings(filePath)

    def _updateView(self, current: QModelIndex, previous: QModelIndex) -> None:
        settingsGroup = self._settingsRegistry[current.row()] if current.isValid() else None
        tableModel = SettingsTableModel(settingsGroup)
        self._tableView.setModel(tableModel)

    def _syncModelToView(self) -> None:
        self._listModel.beginResetModel()
        self._listModel.endResetModel()

        current = self._view.listView.currentIndex()
        self._updateView(current, QModelIndex())

    def update(self, observable: Observable) -> None:
        if observable is self._settingsRegistry:
            self._syncModelToView()

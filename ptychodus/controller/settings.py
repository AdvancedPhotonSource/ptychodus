from __future__ import annotations
from typing import Any

from PyQt5.QtCore import Qt, QAbstractListModel, QAbstractTableModel, QModelIndex, QObject
from PyQt5.QtWidgets import QTableView

from ..api.observer import Observable, Observer
from ..api.settings import SettingsGroup, SettingsRegistry
from ..view.settings import SettingsParametersView
from .data import FileDialogFactory


class SettingsGroupListModel(QAbstractListModel):

    def __init__(self, settingsRegistry: SettingsRegistry, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._settingsRegistry = settingsRegistry

    def refresh(self) -> None:
        self.beginResetModel()
        self.endResetModel()

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if index.isValid() and role == Qt.ItemDataRole.DisplayRole:
            settingsGroup = self._settingsRegistry[index.row()]
            return settingsGroup.name

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._settingsRegistry)


class SettingsEntryTableModel(QAbstractTableModel):

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
            settingsEntry = self._settingsGroup[index.row()]

            if index.column() == 0:
                return settingsEntry.name
            elif index.column() == 1:
                return str(settingsEntry.value)

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._settingsGroup) if self._settingsGroup else 0

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 2


class SettingsController(Observer):

    def __init__(self, settingsRegistry: SettingsRegistry, parametersView: SettingsParametersView,
                 entryTableView: QTableView, fileDialogFactory: FileDialogFactory) -> None:
        super().__init__()
        self._settingsRegistry = settingsRegistry
        self._groupListModel = SettingsGroupListModel(settingsRegistry)
        self._parametersView = parametersView
        self._entryTableView = entryTableView
        self._fileDialogFactory = fileDialogFactory

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry,
                       parametersView: SettingsParametersView, entryTableView: QTableView,
                       fileDialogFactory: FileDialogFactory) -> SettingsController:
        controller = cls(settingsRegistry, parametersView, entryTableView, fileDialogFactory)
        settingsRegistry.addObserver(controller)

        parametersView.settingsView.replacementPathPrefixLineEdit.editingFinished.connect(
            controller._syncReplacementPathPrefixToModel)

        groupListView = parametersView.groupView.listView
        groupListView.setModel(controller._groupListModel)
        groupListView.selectionModel().currentChanged.connect(controller._updateEntryTable)

        parametersView.groupView.buttonBox.openButton.clicked.connect(controller._openSettings)
        parametersView.groupView.buttonBox.saveButton.clicked.connect(controller._saveSettings)

        controller._syncModelToView()

        return controller

    def _syncReplacementPathPrefixToModel(self) -> None:
        self._settingsRegistry.setReplacementPathPrefix(
            self._parametersView.settingsView.replacementPathPrefixLineEdit.text())

    def _openSettings(self) -> None:
        filePath, _ = self._fileDialogFactory.getOpenFilePath(
            self._parametersView,
            'Open Settings',
            nameFilters=self._settingsRegistry.getOpenFileFilterList(),
            selectedNameFilter=self._settingsRegistry.getOpenFileFilter())

        if filePath:
            self._settingsRegistry.openSettings(filePath)

    def _saveSettings(self) -> None:
        filePath, _ = self._fileDialogFactory.getSaveFilePath(
            self._parametersView,
            'Save Settings',
            nameFilters=self._settingsRegistry.getSaveFileFilterList(),
            selectedNameFilter=self._settingsRegistry.getSaveFileFilter())

        if filePath:
            self._settingsRegistry.saveSettings(filePath)

    def _updateEntryTable(self, current: QModelIndex, previous: QModelIndex) -> None:
        settingsGroup = self._settingsRegistry[current.row()] if current.isValid() else None
        entryTableModel = SettingsEntryTableModel(settingsGroup)
        self._entryTableView.setModel(entryTableModel)

    def _syncModelToView(self) -> None:
        replacementPathPrefix = self._settingsRegistry.getReplacementPathPrefix()

        if replacementPathPrefix:
            self._parametersView.settingsView.replacementPathPrefixLineEdit.setText(
                replacementPathPrefix)
        else:
            self._parametersView.settingsView.replacementPathPrefixLineEdit.clear()

        self._groupListModel.refresh()
        current = self._parametersView.groupView.listView.currentIndex()
        self._updateEntryTable(current, QModelIndex())

    def update(self, observable: Observable) -> None:
        if observable is self._settingsRegistry:
            self._syncModelToView()

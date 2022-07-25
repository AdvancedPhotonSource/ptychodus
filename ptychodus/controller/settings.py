from __future__ import annotations
from typing import Optional

from PyQt5.QtCore import (Qt, QAbstractListModel, QAbstractTableModel, QModelIndex, QObject,
                          QVariant)
from PyQt5.QtWidgets import QDialog, QListView, QTableView

from ..api.observer import Observable, Observer
from ..api.settings import SettingsGroup, SettingsRegistry
from ..model import ObjectPresenter, ProbePresenter, ScanPresenter, VelociprobePresenter
from ..view import ImportSettingsDialog
from .data import FileDialogFactory


class SettingsGroupListModel(QAbstractListModel):

    def __init__(self,
                 settingsRegistry: SettingsRegistry,
                 parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._settingsRegistry = settingsRegistry

    def refresh(self) -> None:
        self.beginResetModel()
        self.endResetModel()

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> QVariant:
        value = QVariant()

        if index.isValid() and role == Qt.DisplayRole:
            settingsGroup = self._settingsRegistry[index.row()]
            value = QVariant(settingsGroup.name)

        return value

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._settingsRegistry)


class SettingsEntryTableModel(QAbstractTableModel):

    def __init__(self,
                 settingsGroup: Optional[SettingsGroup],
                 parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._settingsGroup = settingsGroup

    def headerData(self,
                   section: int,
                   orientation: Qt.Orientation,
                   role: int = Qt.DisplayRole) -> QVariant:
        value = QVariant()

        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            if section == 0:
                value = QVariant('Name')
            elif section == 1:
                value = QVariant('Value')

        return value

    def data(self, index: QModelIndex, role: Qt.ItemDataRole) -> QVariant:
        value = QVariant()

        if self._settingsGroup is None:
            return value

        if index.isValid() and role == Qt.DisplayRole:
            settingsEntry = self._settingsGroup[index.row()]

            if index.column() == 0:
                value = QVariant(settingsEntry.name)
            elif index.column() == 1:
                value = QVariant(str(settingsEntry.value))

        return value

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

        groupListView = parametersView.groupView.listView
        groupListView.setModel(controller._groupListModel)
        groupListView.selectionModel().currentChanged.connect(controller._updateEntryTable)

        parametersView.groupView.buttonBox.openButton.clicked.connect(controller._openSettings)
        parametersView.groupView.buttonBox.saveButton.clicked.connect(controller._saveSettings)

        controller._syncModelToView()

        return controller

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
        self._groupListModel.refresh()
        current = self._parametersView.groupView.listView.currentIndex()
        self._updateEntryTable(current, QModelIndex())

    def update(self, observable: Observable) -> None:
        if observable is self._settingsRegistry:
            self._syncModelToView()


class ImportSettingsController(Observer):

    def __init__(self, probePresenter: ProbePresenter, objectPresenter: ObjectPresenter,
                 velociprobePresenter: VelociprobePresenter, dialog: ImportSettingsDialog) -> None:
        super().__init__()
        self._probePresenter = probePresenter
        self._objectPresenter = objectPresenter
        self._velociprobePresenter = velociprobePresenter
        self._dialog = dialog

    @classmethod
    def createInstance(cls, probePresenter: ProbePresenter, objectPresenter: ObjectPresenter,
                       velociprobePresenter: VelociprobePresenter, dialog: ImportSettingsDialog):
        controller = cls(probePresenter, objectPresenter, velociprobePresenter, dialog)
        velociprobePresenter.addObserver(controller)
        dialog.finished.connect(controller._importSettings)
        return controller

    def _importSettings(self, result: int) -> None:
        if result != QDialog.Accepted:
            return

        if self._dialog.valuesGroupBox.detectorPixelCountCheckBox.isChecked():
            self._velociprobePresenter.syncDetectorPixelCount()

        if self._dialog.valuesGroupBox.detectorPixelSizeCheckBox.isChecked():
            self._velociprobePresenter.syncDetectorPixelSize()

        if self._dialog.valuesGroupBox.detectorDistanceCheckBox.isChecked():
            self._velociprobePresenter.syncDetectorDistance()

        self._velociprobePresenter.syncImageCrop(
            syncCenter=self._dialog.valuesGroupBox.imageCropCenterCheckBox.isChecked(),
            syncExtent=self._dialog.valuesGroupBox.imageCropExtentCheckBox.isChecked())

        if self._dialog.valuesGroupBox.probeEnergyCheckBox.isChecked():
            self._velociprobePresenter.syncProbeEnergy()

        if self._dialog.optionsGroupBox.loadScanCheckBox.isChecked():
            self._velociprobePresenter.loadScanFile()

        if self._dialog.optionsGroupBox.reinitializeProbeCheckBox.isChecked():
            self._probePresenter.initializeProbe()

        if self._dialog.optionsGroupBox.reinitializeObjectCheckBox.isChecked():
            self._objectPresenter.initializeObject()

    def update(self, observable: Observable) -> None:
        if observable is self._velociprobePresenter:
            self._dialog.open()

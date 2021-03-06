from __future__ import annotations
from typing import Optional

from PyQt5.QtCore import Qt, QAbstractListModel, QAbstractTableModel, QModelIndex, QObject, QVariant
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

    def data(self, index: QModelIndex, role: Qt.ItemDataRole) -> QVariant:
        value = None

        if index.isValid() and role == Qt.DisplayRole:
            settingsGroup = self._settingsRegistry[index.row()]
            value = settingsGroup.name

        return QVariant(value)

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._settingsRegistry)


class SettingsEntryTableModel(QAbstractTableModel):

    def __init__(self,
                 settingsGroup: Optional[SettingsGroup],
                 parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._settingsGroup = settingsGroup

    def headerData(self, section: int, orientation: Qt.Orientation,
                   role: Qt.ItemDataRole) -> QVariant:
        result = None

        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            if section == 0:
                result = 'Name'
            elif section == 1:
                result = 'Value'

        return QVariant(result)

    def data(self, index: QModelIndex, role: Qt.ItemDataRole) -> QVariant:
        result = None

        if self._settingsGroup is not None and index.isValid() and role == Qt.DisplayRole:
            settingsEntry = self._settingsGroup[index.row()]

            if index.column() == 0:
                result = settingsEntry.name
            elif index.column() == 1:
                result = str(settingsEntry.value)

        return QVariant(result)

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._settingsGroup) if self._settingsGroup else 0

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 2


class SettingsController(Observer):

    def __init__(self, settingsRegistry: SettingsRegistry, groupListView: QListView,
                 entryTableView: QTableView, fileDialogFactory: FileDialogFactory) -> None:
        super().__init__()
        self._settingsRegistry = settingsRegistry
        self._groupListModel = SettingsGroupListModel(settingsRegistry)
        self._groupListView = groupListView
        self._entryTableView = entryTableView
        self._fileDialogFactory = fileDialogFactory

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry, groupListView: QListView,
                       entryTableView: QTableView,
                       fileDialogFactory: FileDialogFactory) -> SettingsController:
        controller = cls(settingsRegistry, groupListView, entryTableView, fileDialogFactory)
        settingsRegistry.addObserver(controller)

        controller._groupListView.setModel(controller._groupListModel)
        controller._groupListView.selectionModel().currentChanged.connect(
            lambda current, previous: controller._updateEntryTable())

        return controller

    def openSettings(self) -> None:
        filePath, _ = self._fileDialogFactory.getOpenFilePath(
            self._groupListView,
            'Open Settings',
            nameFilters=self._settingsRegistry.getOpenFileFilterList(),
            selectedNameFilter=self._settingsRegistry.getOpenFileFilter())

        if filePath:
            self._settingsRegistry.openSettings(filePath)

    def saveSettings(self) -> None:
        filePath, _ = self._fileDialogFactory.getSaveFilePath(
            self._groupListView,
            'Save Settings',
            nameFilters=self._settingsRegistry.getSaveFileFilterList(),
            selectedNameFilter=self._settingsRegistry.getSaveFileFilter())

        if filePath:
            self._settingsRegistry.saveSettings(filePath)

    def _updateEntryTable(self) -> None:
        current = self._groupListView.currentIndex()
        settingsGroup = self._settingsRegistry[current.row()] if current.isValid() else None
        entryTableModel = SettingsEntryTableModel(settingsGroup)
        self._entryTableView.setModel(entryTableModel)

    def update(self, observable: Observable) -> None:
        if observable is self._settingsRegistry:
            self._groupListModel.beginResetModel()
            self._groupListModel.endResetModel()
            self._updateEntryTable()


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

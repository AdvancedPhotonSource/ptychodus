from typing import Optional

from PyQt5.QtCore import Qt, QAbstractListModel, QAbstractTableModel, QModelIndex, QObject, QVariant
from PyQt5.QtWidgets import QDialog, QListView, QTableView

from ..model import ObjectPresenter, Observable, Observer, ProbePresenter, SettingsGroup, SettingsPresenter, SettingsRegistry, VelociprobePresenter
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

        if index.isValid() and role == Qt.DisplayRole:
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
    def __init__(self, settingsRegistry: SettingsRegistry, presenter: SettingsPresenter,
                 groupListView: QListView, entryTableView: QTableView,
                 fileDialogFactory: FileDialogFactory) -> None:
        super().__init__()
        self._settingsRegistry = settingsRegistry
        self._presenter = presenter
        self._groupListModel = SettingsGroupListModel(settingsRegistry)
        self._groupListView = groupListView
        self._entryTableView = entryTableView
        self._fileDialogFactory = fileDialogFactory

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry, presenter: SettingsPresenter,
                       groupListView: QListView, entryTableView: QTableView,
                       fileDialogFactory: FileDialogFactory) -> None:
        controller = cls(settingsRegistry, presenter, groupListView, entryTableView,
                         fileDialogFactory)
        settingsRegistry.addObserver(controller)

        controller._groupListView.setModel(controller._groupListModel)
        controller._groupListView.selectionModel().currentChanged.connect(
            lambda current, previous: controller._updateEntryTable())

        return controller

    def openSettings(self) -> None:
        filePath = self._fileDialogFactory.getOpenFilePath(self._groupListView, 'Open Settings',
                                                           SettingsPresenter.FILE_FILTER)

        if filePath:
            self._presenter.openSettings(filePath)

    def saveSettings(self) -> None:
        filePath = self._fileDialogFactory.getSaveFilePath(self._groupListView, 'Save Settings',
                                                           SettingsPresenter.FILE_FILTER)

        if filePath:
            self._presenter.saveSettings(filePath)

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
            override = self._dialog.optionsGroupBox.fixDetectorDistanceUnitsCheckBox.isChecked()
            self._velociprobePresenter.syncDetectorDistance(override)

        self._velociprobePresenter.syncImageCrop(
            syncCenter=self._dialog.valuesGroupBox.imageCropCenterCheckBox.isChecked(),
            syncExtent=self._dialog.valuesGroupBox.imageCropExtentCheckBox.isChecked())

        if self._dialog.valuesGroupBox.probeEnergyCheckBox.isChecked():
            self._velociprobePresenter.syncProbeEnergy()

        if self._dialog.optionsGroupBox.reinitializeProbeCheckBox.isChecked():
            self._probePresenter.initializeProbe()

        if self._dialog.optionsGroupBox.reinitializeObjectCheckBox.isChecked():
            self._objectPresenter.initializeObject()

    def update(self, observable: Observable) -> None:
        if observable is self._velociprobePresenter:
            self._dialog.open()

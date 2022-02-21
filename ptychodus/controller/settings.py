from pathlib import Path

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from .image import ImageController

from ..model import *
from ..view import ImportSettingsDialog


class SettingsGroupListModel(QAbstractListModel):
    def __init__(self, settingsRegistry: SettingsRegistry, parent: QObject = None) -> None:
        super().__init__(parent)
        self._settingsRegistry = settingsRegistry

    def data(self, index: QModelIndex, role: Qt.ItemDataRole) -> QVariant:
        value = None

        if index.isValid() and role == Qt.DisplayRole:
            settingsGroup = self._settingsRegistry[index.row()]
            value = settingsGroup.name

        return value

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._settingsRegistry)

    def reload(self) -> None:
        self.beginResetModel()
        self.endResetModel()


class SettingsEntryTableModel(QAbstractTableModel):
    def __init__(self, settingsGroup: SettingsGroup, parent: QObject = None) -> None:
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

        return result

    def data(self, index: QModelIndex, role: Qt.ItemDataRole) -> QVariant:
        result = None

        if index.isValid() and role == Qt.DisplayRole:
            settingsEntry = self._settingsGroup[index.row()]

            if index.column() == 0:
                result = settingsEntry.name
            elif index.column() == 1:
                result = str(settingsEntry.value)

        return result

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._settingsGroup)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 2

    def setGroup(self, settingsGroup: SettingsGroup) -> None:
        self.beginResetModel()
        self._settingsGroup = settingsGroup
        self.endResetModel()


class SettingsController(Observer):
    def __init__(self, settingsRegistry: SettingsRegistry, presenter: SettingsPresenter,
                 groupListView: QListView, entryTableView: QTableView) -> None:
        self._settingsRegistry = settingsRegistry
        self._presenter = presenter
        self._groupListModel = SettingsGroupListModel(settingsRegistry)
        self._groupListView = groupListView
        self._entryTableModel = SettingsEntryTableModel(settingsRegistry[0])
        self._entryTableView = entryTableView

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry, presenter: SettingsPresenter,
                       groupListView: QListView, entryTableView: QTableView) -> None:
        controller = cls(settingsRegistry, presenter, groupListView, entryTableView)
        settingsRegistry.addObserver(controller)

        controller._groupListView.setModel(controller._groupListModel)
        controller._entryTableView.setModel(controller._entryTableModel)

        controller._groupListView.selectionModel().currentChanged.connect(
            controller._swapSettingsEntries)

        return controller

    def openSettings(self) -> None:
        fileName, _ = QFileDialog.getOpenFileName(self._groupListView, 'Open Settings',
                                                  str(Path.home()), SettingsPresenter.FILE_FILTER)

        if fileName:
            filePath = Path(fileName)
            self._presenter.openSettings(filePath)

    def saveSettings(self) -> None:
        fileName, _ = QFileDialog.getSaveFileName(self._groupListView, 'Save Settings',
                                                  str(Path.home()), SettingsPresenter.FILE_FILTER)

        if fileName:
            filePath = Path(fileName)
            self._presenter.saveSettings(filePath)

    def _swapSettingsEntries(self, current: QModelIndex, previous: QModelIndex) -> None:
        settingsGroup = self._settingsRegistry[current.row()]
        self._entryTableModel.setGroup(settingsGroup)

    def update(self, observable: Observable) -> None:
        if observable is self._settingsRegistry:
            self._groupListModel.reload()
        # TODO update settingsContentsTableModel


class ImportSettingsController(Observer):
    def __init__(self, presenter: ImportSettingsPresenter, dialog: ImportSettingsDialog) -> None:
        self._presenter = presenter
        self._dialog = dialog

    @classmethod
    def createInstance(cls, presenter: ImportSettingsPresenter, dialog: ImportSettingsDialog):
        controller = cls(presenter, dialog)
        presenter.addObserver(controller)
        dialog.finished.connect(controller._importSettings)
        return controller

    def _importSettings(self, result: int) -> None:
        if result != QDialog.Accepted:
            return

        if self._dialog.detectorPixelSizeCheckBox.isChecked():
            self._presenter.syncDetectorPixelSize()

        if self._dialog.detectorDistanceCheckBox.isChecked():
            self._presenter.syncDetectorDistance()

        if self._dialog.imageCropCenterCheckBox.isChecked():
            self._presenter.syncImageCropCenter()

        # NOTE this must happen after crop center to avoid introducing a bug
        if self._dialog.imageCropExtentCheckBox.isChecked():
            self._presenter.syncImageCropExtent()

        if self._dialog.probeEnergyCheckBox.isChecked():
            self._presenter.syncProbeEnergy()

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._dialog.open()

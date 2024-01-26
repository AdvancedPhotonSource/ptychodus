from __future__ import annotations
import logging
import sys

from PyQt5.QtCore import (Qt, QAbstractTableModel, QItemSelection, QModelIndex, QObject,
                          QSortFilterProxyModel, QVariant)
from PyQt5.QtWidgets import QAbstractItemView

from ...api.observer import SequenceObserver
from ...model.artifact import ArtifactRepositoryPresenter
from ...view.artifact import ArtifactRepositoryView
from ...view.widgets import ExceptionDialog
from ..data import FileDialogFactory
from .info import ArtifactInfoViewController

logger = logging.getLogger(__name__)


class ArtifactRepositoryTableModel(QAbstractTableModel):

    def __init__(self,
                 presenter: ArtifactRepositoryPresenter,
                 parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._presenter = presenter
        self._header = [
            'Name',
            'Probe Energy\n[keV]',
            'Detector-Object\nDistance [m]',
            'Pixel Width\n[nm]',
            'Pixel Height\n[nm]',
            'Size\n[MB]',
        ]

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        value = super().flags(index)

        if index.isValid() and index.column() < 3:
            value |= Qt.ItemFlag.ItemIsEditable

        return value

    def headerData(self,
                   section: int,
                   orientation: Qt.Orientation,
                   role: int = Qt.ItemDataRole.DisplayRole) -> QVariant:
        result = QVariant()

        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            result = QVariant(self._header[section])

        return result

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> QVariant:
        value = QVariant()

        if index.isValid():
            artifact = self._presenter[index.row()]

            if role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.EditRole:
                if index.column() == 0:
                    value = QVariant(artifact.getName())
                elif index.column() == 1:
                    value = QVariant(f'{artifact.getProbeEnergyInElectronVolts() / 1000.:.1f}')
                elif index.column() == 2:
                    value = QVariant(f'{artifact.getDetectorObjectDistanceInMeters():.3g}')
                elif index.column() == 3:
                    value = QVariant('0')  # FIXME objectPlanePixelWidthInMeters
                elif index.column() == 4:
                    value = QVariant('0')  # FIXME objectPlanePixelHeightInMeters
                elif index.column() == 5:
                    value = QVariant(f'{sys.getsizeof(artifact) / (1024 * 1024):.2f}')

        return value

    def setData(self,
                index: QModelIndex,
                value: str,
                role: int = Qt.ItemDataRole.EditRole) -> bool:
        if index.isValid() and role == Qt.ItemDataRole.EditRole:
            artifact = self._presenter[index.row()]

            if index.column() == 0:
                artifact.setName(value)
                self.dataChanged.emit(index, index)
                return True
            elif index.column() == 1:
                try:
                    energyInKiloElectronVolts = float(value)
                except ValueError:
                    pass
                else:
                    artifact.setProbeEnergyInElectronVolts(energyInKiloElectronVolts * 1000)
            elif index.column() == 2:
                try:
                    distanceInMeters = float(value)
                except ValueError:
                    pass
                else:
                    artifact.setDetectorObjectDistanceInMeters(distanceInMeters)

        return False

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._presenter)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._header)


class ArtifactRepositoryController(SequenceObserver):

    def __init__(self, presenter: ArtifactRepositoryPresenter, view: ArtifactRepositoryView,
                 fileDialogFactory: FileDialogFactory, tableModel: ArtifactRepositoryTableModel,
                 tableProxyModel: QSortFilterProxyModel) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view
        self._fileDialogFactory = fileDialogFactory
        self._tableModel = tableModel
        self._tableProxyModel = tableProxyModel

    @classmethod
    def createInstance(cls, presenter: ArtifactRepositoryPresenter, view: ArtifactRepositoryView,
                       fileDialogFactory: FileDialogFactory) -> ArtifactRepositoryController:
        tableModel = ArtifactRepositoryTableModel(presenter)
        tableProxyModel = QSortFilterProxyModel()
        tableProxyModel.setSourceModel(tableModel)

        controller = cls(presenter, view, fileDialogFactory, tableModel, tableProxyModel)
        presenter.addObserver(controller)

        view.tableView.setModel(tableProxyModel)
        view.tableView.setSortingEnabled(True)
        view.tableView.sortByColumn(0, Qt.SortOrder.AscendingOrder)
        view.tableView.verticalHeader().hide()
        view.tableView.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        view.tableView.selectionModel().selectionChanged.connect(controller._updateEnabledButtons)

        openFileAction = view.buttonBox.insertMenu.addAction('Open File...')
        openFileAction.triggered.connect(controller._openArtifact)

        createNewAction = view.buttonBox.insertMenu.addAction('Create New')
        createNewAction.triggered.connect(controller._insertArtifact)

        view.buttonBox.infoButton.clicked.connect(controller._openSelectedArtifactInfo)
        view.buttonBox.saveButton.clicked.connect(controller._saveSelectedArtifact)
        view.buttonBox.removeButton.clicked.connect(controller._removeSelectedArtifact)

        controller._syncModelToView()
        controller._setButtonsEnabled(False)

        return controller

    def _openArtifact(self) -> None:
        filePath, nameFilter = self._fileDialogFactory.getOpenFilePath(
            self._view,
            'Open Artifact',
            nameFilters=self._presenter.getOpenFileFilterList(),
            selectedNameFilter=self._presenter.getOpenFileFilter())

        if filePath:
            self._presenter.openArtifact(filePath, nameFilter)

    def _insertArtifact(self) -> None:
        self._presenter.insertArtifact()

    def _saveSelectedArtifact(self) -> None:
        current = self._tableProxyModel.mapToSource(self._view.tableView.currentIndex())

        if current.isValid():
            filePath, nameFilter = self._fileDialogFactory.getSaveFilePath(
                self._view,
                'Save Artifact',
                nameFilters=self._presenter.getSaveFileFilterList(),
                selectedNameFilter=self._presenter.getSaveFileFilter())

            if filePath:
                try:
                    self._presenter.saveArtifact(current.row(), filePath, nameFilter)
                except Exception as err:
                    logger.exception(err)
                    ExceptionDialog.showException('File Writer', err)
        else:
            logger.error('No items are selected!')

    def _openSelectedArtifactInfo(self) -> None:
        current = self._tableProxyModel.mapToSource(self._view.tableView.currentIndex())

        if current.isValid():
            artifact = self._presenter[current.row()]
            ArtifactInfoViewController.showInfo(artifact, self._view)
        else:
            logger.error('No items are selected!')

    def _removeSelectedArtifact(self) -> None:
        current = self._tableProxyModel.mapToSource(self._view.tableView.currentIndex())

        if current.isValid():
            self._presenter.removeArtifact(current.row())
        else:
            logger.error('No items are selected!')

    def _setButtonsEnabled(self, enabled: bool) -> None:
        self._view.buttonBox.saveButton.setEnabled(enabled)
        self._view.buttonBox.infoButton.setEnabled(enabled)
        self._view.buttonBox.removeButton.setEnabled(enabled)

    def _updateEnabledButtons(self, selected: QItemSelection, deselected: QItemSelection) -> None:
        self._setButtonsEnabled(not selected.isEmpty())

    def _syncModelToView(self) -> None:
        infoText = self._presenter.getInfoText()
        self._view.infoLabel.setText(infoText)

    def handleItemInserted(self, index: int) -> None:
        parent = QModelIndex()
        self._tableModel.beginInsertRows(parent, index, index)
        self._tableModel.endInsertRows()
        self._syncModelToView()

    def handleItemChanged(self, index: int) -> None:
        numberOfColumns = self._tableModel.columnCount()
        topLeft = self._tableModel.index(index, 0)
        bottomRight = self._tableModel.index(index, numberOfColumns - 1)
        self._tableModel.dataChanged.emit(topLeft, bottomRight, [Qt.ItemDataRole.DisplayRole])
        self._syncModelToView()

    def handleItemRemoved(self, index: int) -> None:
        parent = QModelIndex()
        self._tableModel.beginRemoveRows(parent, index, index)
        self._tableModel.endRemoveRows()
        self._syncModelToView()

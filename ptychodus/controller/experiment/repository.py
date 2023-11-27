from __future__ import annotations
import logging

from PyQt5.QtCore import (Qt, QAbstractTableModel, QItemSelection, QModelIndex, QObject,
                          QSortFilterProxyModel, QVariant)
from PyQt5.QtWidgets import QAbstractItemView

from ...api.observer import SequenceObserver
from ...model.experiment import ExperimentRepositoryPresenter
from ...view.experiment import ExperimentRepositoryView
from ...view.widgets import ExceptionDialog
from ..data import FileDialogFactory
from .info import ExperimentInfoViewController

logger = logging.getLogger(__name__)


class ExperimentRepositoryTableModel(QAbstractTableModel):

    def __init__(self,
                 presenter: ExperimentRepositoryPresenter,
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
        value = Qt.ItemFlags()

        if index.isValid():
            value = super().flags(index)

            if index.column() < 3:
                value |= Qt.ItemIsEditable

        return value

    def headerData(self,
                   section: int,
                   orientation: Qt.Orientation,
                   role: int = Qt.DisplayRole) -> QVariant:
        result = QVariant()

        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            result = QVariant(self._header[section])

        return result

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> QVariant:
        value = QVariant()

        if index.isValid():
            experiment = self._presenter[index.row()]

            if role == Qt.DisplayRole or role == Qt.EditRole:
                if index.column() == 0:
                    value = QVariant(experiment.getName())
                elif index.column() == 1:
                    value = QVariant(f'{experiment.getProbeEnergyInElectronVolts() / 1000.:.1f}')
                elif index.column() == 2:
                    value = QVariant(f'{experiment.getDetectorObjectDistanceInMeters():.3g}')
                elif index.column() == 3:
                    value = QVariant('0')  # FIXME objectPlanePixelWidthInMeters
                elif index.column() == 4:
                    value = QVariant('0')  # FIXME objectPlanePixelHeightInMeters
                elif index.column() == 5:
                    value = QVariant(f'{experiment.getSizeInBytes() / (1024 * 1024):.2f}')

        return value

    def setData(self, index: QModelIndex, value: str, role: int = Qt.EditRole) -> bool:
        if index.isValid() and role == Qt.EditRole:
            experiment = self._presenter[index.row()]

            if index.column() == 0:
                experiment.setName(value)
                self.dataChanged.emit(index, index)
                return True
            elif index.column() == 1:
                try:
                    energyInKiloElectronVolts = float(value)
                except ValueError:
                    pass
                else:
                    experiment.setProbeEnergyInElectronVolts(energyInKiloElectronVolts * 1000)
            elif index.column() == 2:
                try:
                    distanceInMeters = float(value)
                except ValueError:
                    pass
                else:
                    experiment.setDetectorObjectDistanceInMeters(distanceInMeters)

        return False

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._presenter)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._header)


class ExperimentRepositoryController(SequenceObserver):

    def __init__(self, presenter: ExperimentRepositoryPresenter, view: ExperimentRepositoryView,
                 fileDialogFactory: FileDialogFactory, tableModel: ExperimentRepositoryTableModel,
                 tableProxyModel: QSortFilterProxyModel) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view
        self._fileDialogFactory = fileDialogFactory
        self._tableModel = tableModel
        self._tableProxyModel = tableProxyModel

    @classmethod
    def createInstance(cls, presenter: ExperimentRepositoryPresenter,
                       view: ExperimentRepositoryView,
                       fileDialogFactory: FileDialogFactory) -> ExperimentRepositoryController:
        tableModel = ExperimentRepositoryTableModel(presenter)
        tableProxyModel = QSortFilterProxyModel()
        tableProxyModel.setSourceModel(tableModel)

        controller = cls(presenter, view, fileDialogFactory, tableModel, tableProxyModel)
        presenter.addObserver(controller)

        view.tableView.setModel(tableProxyModel)
        view.tableView.setSortingEnabled(True)
        view.tableView.sortByColumn(0, Qt.AscendingOrder)
        view.tableView.verticalHeader().hide()
        view.tableView.setSelectionBehavior(QAbstractItemView.SelectRows)
        view.tableView.selectionModel().selectionChanged.connect(controller._updateEnabledButtons)

        openFileAction = view.buttonBox.insertMenu.addAction('Open File...')
        openFileAction.triggered.connect(controller._openExperiment)

        createNewAction = view.buttonBox.insertMenu.addAction('Create New')
        createNewAction.triggered.connect(controller._insertExperiment)

        view.buttonBox.infoButton.clicked.connect(controller._openSelectedExperimentInfo)
        view.buttonBox.saveButton.clicked.connect(controller._saveSelectedExperiment)
        view.buttonBox.removeButton.clicked.connect(controller._removeSelectedExperiment)

        controller._syncModelToView()
        controller._setButtonsEnabled(False)

        return controller

    def _openExperiment(self) -> None:
        filePath, nameFilter = self._fileDialogFactory.getOpenFilePath(
            self._view,
            'Open Experiment',
            nameFilters=self._presenter.getOpenFileFilterList(),
            selectedNameFilter=self._presenter.getOpenFileFilter())

        if filePath:
            self._presenter.openExperiment(filePath, nameFilter)

    def _insertExperiment(self) -> None:
        self._presenter.insertExperiment()

    def _saveSelectedExperiment(self) -> None:
        current = self._tableProxyModel.mapToSource(self._view.tableView.currentIndex())

        if current.isValid():
            filePath, nameFilter = self._fileDialogFactory.getSaveFilePath(
                self._view,
                'Save Experiment',
                nameFilters=self._presenter.getSaveFileFilterList(),
                selectedNameFilter=self._presenter.getSaveFileFilter())

            if filePath:
                try:
                    self._presenter.saveExperiment(current.row(), filePath, nameFilter)
                except Exception as err:
                    logger.exception(err)
                    ExceptionDialog.showException('File Writer', err)
        else:
            logger.error('No items are selected!')

    def _openSelectedExperimentInfo(self) -> None:
        current = self._tableProxyModel.mapToSource(self._view.tableView.currentIndex())

        if current.isValid():
            experiment = self._presenter[current.row()]
            ExperimentInfoViewController.showInfo(experiment, self._view)
        else:
            logger.error('No items are selected!')

    def _removeSelectedExperiment(self) -> None:
        current = self._tableProxyModel.mapToSource(self._view.tableView.currentIndex())

        if current.isValid():
            self._presenter.removeExperiment(current.row())
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
        self._tableModel.dataChanged.emit(topLeft, bottomRight, [Qt.DisplayRole])
        self._syncModelToView()

    def handleItemRemoved(self, index: int) -> None:
        parent = QModelIndex()
        self._tableModel.beginRemoveRows(parent, index, index)
        self._tableModel.endRemoveRows()
        self._syncModelToView()

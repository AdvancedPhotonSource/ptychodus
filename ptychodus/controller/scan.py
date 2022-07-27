from __future__ import annotations
from decimal import Decimal
from typing import Callable
import logging

from PyQt5.QtCore import (Qt, QAbstractTableModel, QModelIndex, QObject, QSortFilterProxyModel,
                          QVariant)
from PyQt5.QtWidgets import QAbstractItemView

from ..api.observer import Observer, Observable
from ..model import Scan, ScanPresenter, ScanRepositoryEntry
from ..view import ScanPositionDataView, ScanPlotView, ScanEditorView, ScanTransformView
from .data import FileDialogFactory

logger = logging.getLogger(__name__)


class ScanEditorController(Observer):

    def __init__(self, presenter: ScanPresenter, view: ScanEditorView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

    @classmethod
    def createInstance(cls, presenter: ScanPresenter,
                       view: ScanEditorView) -> ScanEditorController:
        controller = cls(presenter, view)
        presenter.addObserver(controller)

        view.numberOfScanPointsSpinBox.setEnabled(False)
        view.extentXSpinBox.valueChanged.connect(presenter.setExtentX)
        view.extentYSpinBox.valueChanged.connect(presenter.setExtentY)

        view.stepSizeXWidget.lengthChanged.connect(presenter.setStepSizeXInMeters)
        view.stepSizeYWidget.lengthChanged.connect(presenter.setStepSizeYInMeters)

        controller._syncModelToView()

        return controller

    def _syncModelToView(self) -> None:
        self._view.numberOfScanPointsSpinBox.blockSignals(True)
        self._view.numberOfScanPointsSpinBox.setRange(
            self._presenter.getNumberOfScanPointsLimits().lower,
            self._presenter.getNumberOfScanPointsLimits().upper)
        self._view.numberOfScanPointsSpinBox.setValue(self._presenter.getNumberOfScanPoints())
        self._view.numberOfScanPointsSpinBox.blockSignals(False)

        self._view.extentXSpinBox.blockSignals(True)
        self._view.extentXSpinBox.setRange(self._presenter.getExtentXLimits().lower,
                                           self._presenter.getExtentXLimits().upper)
        self._view.extentXSpinBox.setValue(self._presenter.getExtentX())
        self._view.extentXSpinBox.blockSignals(False)

        self._view.extentYSpinBox.blockSignals(True)
        self._view.extentYSpinBox.setRange(self._presenter.getExtentYLimits().lower,
                                           self._presenter.getExtentYLimits().upper)
        self._view.extentYSpinBox.setValue(self._presenter.getExtentY())
        self._view.extentYSpinBox.blockSignals(False)

        self._view.stepSizeXWidget.setLengthInMeters(self._presenter.getStepSizeXInMeters())
        self._view.stepSizeYWidget.setLengthInMeters(self._presenter.getStepSizeYInMeters())

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class ScanTransformController(Observer):

    def __init__(self, presenter: ScanPresenter, view: ScanTransformView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

    @classmethod
    def createInstance(cls, presenter: ScanPresenter,
                       view: ScanTransformView) -> ScanTransformController:
        controller = cls(presenter, view)
        presenter.addObserver(controller)

        for transform in presenter.getTransformList():
            view.transformComboBox.addItem(transform)

        view.transformComboBox.currentTextChanged.connect(presenter.setTransform)
        view.jitterRadiusWidget.lengthChanged.connect(presenter.setJitterRadiusInMeters)

        controller._syncModelToView()

        return controller

    def _syncModelToView(self) -> None:
        self._view.transformComboBox.setCurrentText(self._presenter.getTransform())
        self._view.jitterRadiusWidget.setLengthInMeters(self._presenter.getJitterRadiusInMeters())

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class ScanTableModel(QAbstractTableModel):

    def __init__(self, presenter: ScanPresenter, parent: QObject = None) -> None:
        super().__init__(parent)
        self._presenter = presenter
        self._scanList: list[ScanRepositoryEntry] = list()
        self._checkedNames: set[str] = set()

    def refresh(self) -> None:
        scanList = self._presenter.getScanRepositoryContents()

        if len(self._scanList) == len(scanList):
            topLeft = self.index(0, 0)
            bottomRight = self.index(len(self._scanList), self.columnCount())
            self.dataChanged.emit(topLeft, bottomRight)
        else:
            self.beginResetModel()
            self._scanList = scanList
            self._scanList.sort(key=lambda entry: entry.name)
            self.endResetModel()

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        value = super().flags(index)

        if index.isValid():
            entry = self._scanList[index.row()]

            if entry.variant == 'FromMemory':
                value &= ~Qt.ItemIsSelectable

            if index.column() == 0:
                value |= Qt.ItemIsUserCheckable

        return value

    def headerData(self,
                   section: int,
                   orientation: Qt.Orientation,
                   role: int = Qt.DisplayRole) -> QVariant:
        result = QVariant()

        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            if section == 0:
                result = QVariant('Name')
            elif section == 1:
                result = QVariant('Category')
            elif section == 2:
                result = QVariant('Variant')
            elif section == 3:
                result = QVariant('Length')

        return result

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> QVariant:
        value = QVariant()

        if index.isValid():
            entry = self._scanList[index.row()]

            if role == Qt.CheckStateRole:
                if index.column() == 0:
                    value = QVariant(Qt.Checked if entry.name in
                                     self._checkedNames else Qt.Unchecked)
            elif role == Qt.DisplayRole:
                if index.column() == 0:
                    value = QVariant(entry.name)
                elif index.column() == 1:
                    value = QVariant(entry.category)
                elif index.column() == 2:
                    value = QVariant(entry.variant)
                elif index.column() == 3:
                    value = QVariant(len(entry.pointSequence))

        return value

    def setData(self, index: QModelIndex, value: QVariant, role: int = Qt.EditRole) -> bool:
        if index.isValid() and index.column() == 0 and role == Qt.CheckStateRole:
            entry = self._scanList[index.row()]

            if value == Qt.Checked:
                self._checkedNames.add(entry.name)
            else:
                self._checkedNames.discard(entry.name)

            self.dataChanged.emit(index, index)

            return True

        return False

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._scanList)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 4


class ScanPositionDataController(Observer):

    def __init__(self, presenter: ScanPresenter, view: ScanPositionDataView,
                 fileDialogFactory: FileDialogFactory) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view
        self._fileDialogFactory = fileDialogFactory
        self._tableModel = ScanTableModel(presenter)
        self._proxyModel = QSortFilterProxyModel()

    @classmethod
    def createInstance(cls, presenter: ScanPresenter, view: ScanPositionDataView,
                       fileDialogFactory: FileDialogFactory) -> ScanPositionDataController:
        controller = cls(presenter, view, fileDialogFactory)
        presenter.addObserver(controller)

        controller._proxyModel.setSourceModel(controller._tableModel)

        view.tableView.setModel(controller._proxyModel)
        view.tableView.setSortingEnabled(True)
        view.tableView.setSelectionBehavior(QAbstractItemView.SelectRows)
        view.tableView.selectionModel().currentChanged.connect(controller._setActiveScan)

        openFileAction = view.buttonBox.insertMenu.addAction('Open File...')
        openFileAction.triggered.connect(lambda checked: controller._openScan())

        for name in presenter.getInitializerNameList():
            insertAction = view.buttonBox.insertMenu.addAction(name)
            insertAction.triggered.connect(controller._createInitLambda(name))

        view.buttonBox.editButton.clicked.connect(controller._editSelectedScan)
        view.buttonBox.saveButton.clicked.connect(controller._saveSelectedScan)
        view.buttonBox.removeButton.clicked.connect(controller._removeSelectedScan)

        controller._syncModelToView()

        return controller

    def _createInitLambda(self, name: str) -> Callable[[bool], None]:
        # NOTE additional defining scope for lambda forces a new instance for each use
        return lambda checked: self._presenter.initializeScan(name)

    def _openScan(self) -> None:
        filePath, nameFilter = self._fileDialogFactory.getOpenFilePath(
            self._view,
            'Open Scan',
            nameFilters=self._presenter.getOpenFileFilterList(),
            selectedNameFilter=self._presenter.getOpenFileFilter())

        if filePath:
            self._presenter.openScan(filePath, nameFilter)

    def _saveSelectedScan(self) -> None:
        current = self._view.tableView.selectionModel().currentIndex()

        if not current.isValid():
            logger.error('No scans are selected!')
            return

        filePath, nameFilter = self._fileDialogFactory.getSaveFilePath(
            self._view,
            'Save Scan',
            nameFilters=self._presenter.getSaveFileFilterList(),
            selectedNameFilter=self._presenter.getSaveFileFilter())

        if filePath:
            name = current.sibling(current.row(), 0).data()
            self._presenter.saveScan(filePath, nameFilter, name)

    def _editSelectedScan(self) -> None:
        current = self._view.tableView.selectionModel().currentIndex()

        if current.isValid():
            name = current.sibling(current.row(), 0).data()
            logger.debug(f'editScan({name})')  # FIXME implement
        else:
            logger.error('No scans are selected!')

    def _removeSelectedScan(self) -> None:
        current = self._view.tableView.selectionModel().currentIndex()

        if current.isValid():
            name = current.sibling(current.row(), 0).data()
            self._presenter.removeScan(name)
        else:
            logger.error('No scans are selected!')

    def _setActiveScan(self, current: QModelIndex, previous: QModelIndex) -> None:
        name = current.sibling(current.row(), 0).data()
        self._presenter.setActiveScan(name)

    def _syncModelToView(self) -> None:
        self._tableModel.refresh()

        for row in range(self._proxyModel.rowCount()):
            index = self._proxyModel.index(row, 0)
            name = index.data()

            if name == self._presenter.getActiveScan():
                self._view.tableView.selectRow(row)
                return

        self._view.tableView.clearSelection()

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class ScanPlotController(Observer):

    def __init__(self, presenter: ScanPresenter, view: ScanPlotView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

    @classmethod
    def createInstance(cls, presenter: ScanPresenter, view: ScanPlotView) -> ScanPlotController:
        controller = cls(presenter, view)
        presenter.addObserver(controller)
        controller._syncModelToView()
        return controller

    def _syncModelToView(self) -> None:
        scanPath = self._presenter.getActiveScanPointList()

        x = [point.x for point in scanPath]
        y = [point.y for point in scanPath]

        self._view.axes.clear()
        self._view.axes.plot(x, y, '.-', linewidth=1.5)
        self._view.axes.invert_yaxis()
        self._view.axes.axis('equal')
        self._view.axes.grid(True)
        self._view.axes.set_xlabel('X [m]')
        self._view.axes.set_ylabel('Y [m]')
        self._view.figureCanvas.draw()

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()

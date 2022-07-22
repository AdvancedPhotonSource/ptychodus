from __future__ import annotations
from decimal import Decimal

from PyQt5.QtCore import Qt, QAbstractTableModel, QModelIndex, QObject, QVariant
from PyQt5.QtWidgets import QAbstractItemView

from ..api.observer import Observer, Observable
from ..model import Scan, ScanPresenter, ScanRepositoryEntry
from ..view import ScanParametersView, ScanPlotView, ScanEditorView, ScanTransformView
from .data import FileDialogFactory


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

    def getName(self, index: QModelIndex) -> str:
        return self._scanList[index.row()].name

    def refresh(self) -> None:
        # TODO emit dataChanged appropriately
        self.beginResetModel()
        self._scanList = self._presenter.getScanRepositoryContents()
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

    def headerData(self, section: int, orientation: Qt.Orientation, role: int) -> QVariant:
        result = QVariant()

        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            if section == 0:
                result = 'Name'
            elif section == 1:
                result = 'Category'
            elif section == 2:
                result = 'Variant'
            elif section == 3:
                result = 'Length'

        return result

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> QVariant:
        value = QVariant()

        if index.isValid():
            entry = self._scanList[index.row()]

            if role == Qt.CheckStateRole:
                if index.column() == 0:
                    value = Qt.Checked if entry.name in self._checkedNames else Qt.Unchecked
            elif role == Qt.DisplayRole:
                if index.column() == 0:
                    value = entry.name
                elif index.column() == 1:
                    value = entry.category
                elif index.column() == 2:
                    value = entry.variant
                elif index.column() == 3:
                    value = len(entry.pointSequence)

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


class ScanParametersController(Observer):

    def __init__(self, presenter: ScanPresenter, view: ScanParametersView,
                 fileDialogFactory: FileDialogFactory) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view
        self._fileDialogFactory = fileDialogFactory
        self._tableModel = ScanTableModel(presenter) # TODO update when presenter changes

    @classmethod
    def createInstance(cls, presenter: ScanPresenter, view: ScanParametersView,
                       fileDialogFactory: FileDialogFactory) -> ScanParametersController:
        controller = cls(presenter, view, fileDialogFactory)
        presenter.addObserver(controller)

        view.tableView.setModel(controller._tableModel)
        view.tableView.setSelectionBehavior(QAbstractItemView.SelectRows)
        view.tableView.selectionModel().currentChanged.connect(controller._setActiveScan)

        openFileAction = view.buttonBox.insertMenu.addAction('Open File...')
        openFileAction.triggered.connect(lambda checked: controller._openScan())

        insertRasterAction = view.buttonBox.insertMenu.addAction('Raster')
        insertRasterAction.triggered.connect(lambda checked: presenter.insertRasterScan())

        insertSnakeAction = view.buttonBox.insertMenu.addAction('Snake')
        insertSnakeAction.triggered.connect(lambda checked: presenter.insertSnakeScan())

        insertSpiralAction = view.buttonBox.insertMenu.addAction('Spiral')
        insertSpiralAction.triggered.connect(lambda checked: presenter.insertSpiralScan())

        view.buttonBox.editButton.clicked.connect(controller._editSelectedScan)
        view.buttonBox.saveButton.clicked.connect(controller._saveSelectedScan)
        view.buttonBox.removeButton.clicked.connect(controller._removeSelectedScan)

        # FIXME tableView selected/checked

        controller._syncModelToView()

        return controller

    def _openScan(self) -> None:
        filePath, nameFilter = self._fileDialogFactory.getOpenFilePath(
            self._view,
            'Open Scan',
            nameFilters=self._presenter.getOpenFileFilterList(),
            selectedNameFilter=self._presenter.getOpenFileFilter())

        if filePath:
            self._presenter.openScan(filePath, nameFilter)

    def _getSelectedScan(self) -> str:
        # FIXME handle case of no selection
        current = self._view.tableView.selectionModel().currentIndex()
        return self._tableModel.getName(current)

    def _saveSelectedScan(self) -> None:
        filePath, nameFilter = self._fileDialogFactory.getSaveFilePath(
            self._view,
            'Save Scan',
            nameFilters=self._presenter.getSaveFileFilterList(),
            selectedNameFilter=self._presenter.getSaveFileFilter())

        if filePath:
            name = self._getSelectedScan()
            self._presenter.saveScan(filePath, nameFilter, name)

    def _editSelectedScan(self) -> None:
        name = self._getSelectedScan()
        print(f'editScan({name})')  # FIXME

    def _removeSelectedScan(self) -> None:
        name = self._getSelectedScan()
        self._presenter.removeScan(name)

    def _setActiveScan(self, current: QModelIndex, previous: QModelIndex) -> None:
        name = self._tableModel.getName(current)
        self._presenter.setActiveScan(name)

    def _syncModelToView(self) -> None:
        self._tableModel.refresh()
        # FIXME presenter.getActiveScan() to update tableView selection

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

from __future__ import annotations
from typing import Callable
import logging

from PyQt5.QtCore import (Qt, QAbstractTableModel, QModelIndex, QObject, QSortFilterProxyModel,
                          QVariant)
from PyQt5.QtWidgets import QAbstractItemView

from ...api.observer import Observable, Observer
from ...model import (CartesianScanInitializer, Scan, ScanInitializer, ScanPresenter,
                      SpiralScanInitializer)
from ...view import (ScanEditorDialog, ScanParametersView, ScanPlotView, ScanPositionDataView,
                     ScanTransformView)
from ..data import FileDialogFactory
from .cartesian import CartesianScanController
from .spiral import SpiralScanController
from .tableModel import ScanTableModel
from .transformController import ScanTransformController

logger = logging.getLogger(__name__)


class ScanController(Observer):
    OPEN_FILE = 'Open File...'

    def __init__(self, presenter: ScanPresenter, parametersView: ScanParametersView,
                 plotView: ScanPlotView, fileDialogFactory: FileDialogFactory) -> None:
        super().__init__()
        self._presenter = presenter
        self._parametersView = parametersView
        self._plotView = plotView
        self._fileDialogFactory = fileDialogFactory
        self._tableModel = ScanTableModel(presenter)
        self._proxyModel = QSortFilterProxyModel()

    @classmethod
    def createInstance(cls, presenter: ScanPresenter, parametersView: ScanParametersView,
                       plotView: ScanPlotView,
                       fileDialogFactory: FileDialogFactory) -> ScanController:
        controller = cls(presenter, parametersView, plotView, fileDialogFactory)
        presenter.addObserver(controller)

        controller._proxyModel.setSourceModel(controller._tableModel)

        parametersView.positionDataView.tableView.setModel(controller._proxyModel)
        parametersView.positionDataView.tableView.setSortingEnabled(True)
        parametersView.positionDataView.tableView.setSelectionBehavior(
            QAbstractItemView.SelectRows)
        parametersView.positionDataView.tableView.setSelectionMode(
            QAbstractItemView.SingleSelection)
        parametersView.positionDataView.tableView.activated.connect(controller._setActiveScan)
        parametersView.positionDataView.tableView.selectionModel().selectionChanged.connect(
            lambda selected, deselected: controller._setButtonsEnabled())
        parametersView.positionDataView.tableView.horizontalHeader().sectionClicked.connect(
            lambda logicalIndex: controller._redrawPlot())

        initializerNameList = presenter.getInitializerNameList()
        initializerNameList.insert(0, ScanController.OPEN_FILE)

        for name in initializerNameList:
            insertAction = parametersView.positionDataView.buttonBox.insertMenu.addAction(name)
            insertAction.triggered.connect(controller._createInitializerLambda(name))

        parametersView.positionDataView.buttonBox.editButton.clicked.connect(
            controller._editSelectedScan)
        parametersView.positionDataView.buttonBox.saveButton.clicked.connect(
            controller._saveSelectedScan)
        parametersView.positionDataView.buttonBox.removeButton.clicked.connect(
            controller._removeSelectedScan)

        controller._proxyModel.dataChanged.connect(
            lambda topLeft, bottomRight, roles: controller._redrawPlot())
        controller._syncModelToView()

        return controller

    def _initializeScan(self, name: str) -> None:
        if name == ScanController.OPEN_FILE:
            self._openScan()
        else:
            self._presenter.initializeScan(name)

    def _createInitializerLambda(self, name: str) -> Callable[[bool], None]:
        # NOTE additional defining scope for lambda forces a new instance for each use
        return lambda checked: self._initializeScan(name)

    def _openScan(self) -> None:
        filePath, nameFilter = self._fileDialogFactory.getOpenFilePath(
            self._parametersView.positionDataView,
            'Open Scan',
            nameFilters=self._presenter.getOpenFileFilterList(),
            selectedNameFilter=self._presenter.getOpenFileFilter())

        if filePath:
            self._presenter.openScan(filePath, nameFilter)

    def _saveSelectedScan(self) -> None:
        current = self._parametersView.positionDataView.tableView.selectionModel().currentIndex()

        if not current.isValid():
            logger.error('No scans are selected!')
            return

        filePath, nameFilter = self._fileDialogFactory.getSaveFilePath(
            self._parametersView.positionDataView,
            'Save Scan',
            nameFilters=self._presenter.getSaveFileFilterList(),
            selectedNameFilter=self._presenter.getSaveFileFilter())

        if filePath:
            name = current.sibling(current.row(), 0).data()
            self._presenter.saveScan(filePath, nameFilter, name)

    def _editSelectedScan(self) -> None:
        current = self._parametersView.positionDataView.tableView.selectionModel().currentIndex()

        if current.isValid():
            name = current.sibling(current.row(), 0).data()
            category = current.sibling(current.row(), 1).data()
            initializer = self._presenter.getInitializer(name)

            isCartesian = (category.casefold() == 'cartesian')
            dialog = ScanEditorDialog.createInstance(isCartesian, self._parametersView)
            dialog.setWindowTitle(name)

            if isinstance(initializer, CartesianScanInitializer):
                cartesianController = CartesianScanController.createInstance(
                    initializer, dialog.editorView)
            elif isinstance(initializer, SpiralScanInitializer):
                spiralController = SpiralScanController.createInstance(
                    initializer, dialog.editorView)
            else:
                logger.debug(f'Unknown category \"{category}\"')

            transformController = ScanTransformController.createInstance(
                initializer, dialog.transformView)

            dialog.open()
        else:
            logger.error('No scans are selected!')

    def _removeSelectedScan(self) -> None:
        current = self._parametersView.positionDataView.tableView.selectionModel().currentIndex()

        if current.isValid():
            name = current.sibling(current.row(), 0).data()
            self._presenter.removeScan(name)
        else:
            logger.error('No scans are selected!')

    def _setButtonsEnabled(self) -> None:
        selectionModel = self._parametersView.positionDataView.tableView.selectionModel()
        enable = False
        enableRemove = False

        for index in selectionModel.selectedIndexes():
            if index.isValid():
                enable = True
                name = index.sibling(index.row(), 0).data()
                enableRemove |= self._presenter.canRemoveScan(name)

        self._parametersView.positionDataView.buttonBox.saveButton.setEnabled(enable)
        self._parametersView.positionDataView.buttonBox.editButton.setEnabled(enable)
        self._parametersView.positionDataView.buttonBox.removeButton.setEnabled(enableRemove)

    def _setActiveScan(self, index: QModelIndex) -> None:
        name = index.sibling(index.row(), 0).data()
        self._presenter.setActiveScan(name)

    def _redrawPlot(self) -> None:
        initializerDict = {
            name: initializer
            for name, initializer in self._presenter.getScanRepositoryContents()
        }
        self._plotView.axes.clear()

        for row in range(self._proxyModel.rowCount()):
            index = self._proxyModel.index(row, 0)

            if index.data(Qt.CheckStateRole) != Qt.Checked:
                continue

            name = index.data()
            initializer = initializerDict[name]
            x = [point.x for point in initializer]
            y = [point.y for point in initializer]

            self._plotView.axes.plot(x, y, '.-', label=name, linewidth=1.5)

        self._plotView.axes.invert_yaxis()
        self._plotView.axes.axis('equal')
        self._plotView.axes.grid(True)
        self._plotView.axes.set_xlabel('X [m]')
        self._plotView.axes.set_ylabel('Y [m]')

        if len(self._plotView.axes.lines) > 0:
            self._plotView.axes.legend(loc='best')

        self._plotView.figureCanvas.draw()

    def _syncModelToView(self) -> None:
        for _, initializer in self._presenter.getScanRepositoryContents():
            initializer.addObserver(self)

        self._tableModel.refresh()
        self._setButtonsEnabled()
        self._redrawPlot()

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()
        else:
            for name, initializer in self._presenter.getScanRepositoryContents():
                if observable is initializer and self._tableModel.isChecked(name):
                    self._redrawPlot()
                    break

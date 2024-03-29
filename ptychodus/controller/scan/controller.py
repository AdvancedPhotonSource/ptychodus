from __future__ import annotations
from typing import Callable, Final
import logging

from PyQt5.QtCore import Qt, QSortFilterProxyModel
from PyQt5.QtWidgets import QAbstractItemView

from ...api.observer import Observable, Observer
from ...model.scan import (ScanRepositoryItem, ScanRepositoryItemPresenter,
                           ScanRepositoryPresenter)
from ...view.scan import ScanView, ScanPlotView
from ...view.widgets import ExceptionDialog
from ..data import FileDialogFactory
from .cartesian import CartesianScanController
from .concentric import ConcentricScanController
from .lissajous import LissajousScanController
from .spiral import SpiralScanController
from .tableModel import ScanTableModel
from .tabular import TabularScanController

logger = logging.getLogger(__name__)


class ScanController(Observer):
    OPEN_FILE: Final[str] = 'Open File...'  # TODO clean up

    def __init__(self, repositoryPresenter: ScanRepositoryPresenter, view: ScanView,
                 plotView: ScanPlotView, fileDialogFactory: FileDialogFactory) -> None:
        super().__init__()
        self._repositoryPresenter = repositoryPresenter
        self._view = view
        self._plotView = plotView
        self._fileDialogFactory = fileDialogFactory
        self._tableModel = ScanTableModel(repositoryPresenter)
        self._proxyModel = QSortFilterProxyModel()

    @classmethod
    def createInstance(cls, repositoryPresenter: ScanRepositoryPresenter, view: ScanView,
                       plotView: ScanPlotView,
                       fileDialogFactory: FileDialogFactory) -> ScanController:
        controller = cls(repositoryPresenter, view, plotView, fileDialogFactory)
        repositoryPresenter.addObserver(controller)

        controller._proxyModel.setSourceModel(controller._tableModel)
        view.repositoryView.tableView.setModel(controller._proxyModel)
        view.repositoryView.tableView.setSortingEnabled(True)
        view.repositoryView.tableView.setSelectionBehavior(QAbstractItemView.SelectRows)
        view.repositoryView.tableView.setSelectionMode(QAbstractItemView.SingleSelection)
        view.repositoryView.tableView.selectionModel().selectionChanged.connect(
            controller._updateView)
        view.repositoryView.tableView.horizontalHeader().sectionClicked.connect(
            lambda logicalIndex: controller._redrawPlot())

        for name in repositoryPresenter.getInitializerDisplayNameList():
            insertAction = view.repositoryView.buttonBox.insertMenu.addAction(name)
            insertAction.triggered.connect(controller._createItemLambda(name))

        view.repositoryView.buttonBox.editButton.clicked.connect(controller._editSelectedScan)
        view.repositoryView.buttonBox.saveButton.clicked.connect(controller._saveSelectedScan)
        view.repositoryView.buttonBox.removeButton.clicked.connect(controller._removeSelectedScan)

        controller._proxyModel.dataChanged.connect(
            lambda topLeft, bottomRight, roles: controller._redrawPlot())
        controller._syncModelToView()

        return controller

    def _initializeScan(self, name: str) -> None:
        if name == ScanController.OPEN_FILE:
            self._openScan()
        else:
            self._repositoryPresenter.initializeScan(name)

    def _createItemLambda(self, name: str) -> Callable[[bool], None]:
        # NOTE additional defining scope for lambda forces a new instance for each use
        return lambda checked: self._initializeScan(name)

    def _openScan(self) -> None:
        filePath, nameFilter = self._fileDialogFactory.getOpenFilePath(
            self._view.repositoryView,
            'Open Scan',
            nameFilters=self._repositoryPresenter.getOpenFileFilterList(),
            selectedNameFilter=self._repositoryPresenter.getOpenFileFilter())

        if filePath:
            self._repositoryPresenter.openScan(filePath, nameFilter)

    def _getCurrentItemPresenter(self) -> ScanRepositoryItemPresenter | None:
        itemPresenter: ScanRepositoryItemPresenter | None = None
        proxyIndex = self._view.repositoryView.tableView.currentIndex()

        if proxyIndex.isValid():
            index = self._proxyModel.mapToSource(proxyIndex)
            itemPresenter = self._repositoryPresenter[index.row()]
        else:
            logger.error('No items are selected!')

        return itemPresenter

    def _saveSelectedScan(self) -> None:
        itemPresenter = self._getCurrentItemPresenter()

        if itemPresenter is not None:
            filePath, nameFilter = self._fileDialogFactory.getSaveFilePath(
                self._view.repositoryView,
                'Save Scan',
                nameFilters=self._repositoryPresenter.getSaveFileFilterList(),
                selectedNameFilter=self._repositoryPresenter.getSaveFileFilter())

            if filePath:
                try:
                    self._repositoryPresenter.saveScan(itemPresenter.name, filePath, nameFilter)
                except Exception as err:
                    logger.exception(err)
                    ExceptionDialog.showException('File writer', err)

    def _editSelectedScan(self) -> None:
        itemPresenter = self._getCurrentItemPresenter()

        if itemPresenter is not None:
            item = itemPresenter.item
            initializerName = item.getInitializerSimpleName()

            if initializerName in ('Snake', 'Raster', 'CenteredSnake', 'CenteredRaster'):
                CartesianScanController.editParameters(itemPresenter, self._view)
            elif initializerName == 'Concentric':
                ConcentricScanController.editParameters(itemPresenter, self._view)
            elif initializerName == 'Spiral':
                SpiralScanController.editParameters(itemPresenter, self._view)
            elif initializerName == 'Lissajous':
                LissajousScanController.editParameters(itemPresenter, self._view)
            else:
                TabularScanController.editParameters(itemPresenter, self._view)

    def _removeSelectedScan(self) -> None:
        itemPresenter = self._getCurrentItemPresenter()

        if itemPresenter is not None:
            self._repositoryPresenter.removeScan(itemPresenter.name)

    def _redrawPlot(self) -> None:
        itemDict = {
            itemPresenter.name: itemPresenter.item
            for itemPresenter in self._repositoryPresenter
        }
        self._plotView.axes.clear()

        for row in range(self._proxyModel.rowCount()):
            index = self._proxyModel.index(row, 0)

            if index.data(Qt.CheckStateRole) != Qt.Checked:
                continue

            name = index.data()
            item = itemDict[name]
            x = [point.x for point in item.values()]
            y = [point.y for point in item.values()]

            self._plotView.axes.plot(x, y, '.-', label=name, linewidth=1.5)

        self._plotView.axes.invert_yaxis()
        self._plotView.axes.axis('equal')
        self._plotView.axes.grid(True)
        self._plotView.axes.set_xlabel('X [m]')
        self._plotView.axes.set_ylabel('Y [m]')

        if len(self._plotView.axes.lines) > 0:
            self._plotView.axes.legend(loc='best')

        self._plotView.figureCanvas.draw()

    def _setButtonsEnabled(self) -> None:
        selectionModel = self._view.repositoryView.tableView.selectionModel()
        hasSelection = selectionModel.hasSelection()

        self._view.repositoryView.buttonBox.saveButton.setEnabled(hasSelection)
        self._view.repositoryView.buttonBox.editButton.setEnabled(hasSelection)
        self._view.repositoryView.buttonBox.removeButton.setEnabled(hasSelection)

    def _updateView(self) -> None:
        self._setButtonsEnabled()
        self._redrawPlot()

    def _syncModelToView(self) -> None:
        for itemPresenter in self._repositoryPresenter:
            itemPresenter.item.addObserver(self)

        self._tableModel.beginResetModel()
        self._tableModel.endResetModel()

    def update(self, observable: Observable) -> None:
        if observable is self._repositoryPresenter:
            self._syncModelToView()
        elif isinstance(observable, ScanRepositoryItem):
            for itemPresenter in self._repositoryPresenter:
                if observable is itemPresenter.item:
                    if self._tableModel.isChecked(itemPresenter.name):
                        self._redrawPlot()

                    break

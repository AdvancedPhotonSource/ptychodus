from __future__ import annotations
from typing import Callable, Final, Optional
import logging

from PyQt5.QtCore import (Qt, QAbstractTableModel, QModelIndex, QObject, QSortFilterProxyModel,
                          QVariant)
from PyQt5.QtWidgets import QAbstractItemView

from ...api.observer import Observable, Observer
from ...model.scan import (CartesianScanRepositoryItem, LissajousScanRepositoryItem,
                           ScanRepositoryItemPresenter, ScanRepositoryPresenter,
                           SpiralScanRepositoryItem, TabularScanRepositoryItem,
                           TransformedScanRepositoryItem)
from ...view import (CartesianScanView, LissajousScanView, RepositoryWidget, ScanEditorDialog,
                     ScanParametersView, ScanPlotView, ScanTransformView, SpiralScanView,
                     TabularScanView)
from ..data import FileDialogFactory
from .cartesian import CartesianScanController
from .lissajous import LissajousScanController
from .spiral import SpiralScanController
from .tableModel import ScanTableModel
from .transformController import ScanTransformController

logger = logging.getLogger(__name__)


class ScanController(Observer):
    OPEN_FILE: Final[str] = 'Open File...'

    def __init__(self, repositoryPresenter: ScanRepositoryPresenter,
                 parametersView: ScanParametersView, plotView: ScanPlotView,
                 fileDialogFactory: FileDialogFactory) -> None:
        super().__init__()
        self._repositoryPresenter = repositoryPresenter
        self._parametersView = parametersView
        self._plotView = plotView
        self._fileDialogFactory = fileDialogFactory
        self._tableModel = ScanTableModel(repositoryPresenter)
        self._proxyModel = QSortFilterProxyModel()

    @classmethod
    def createInstance(cls, repositoryPresenter: ScanRepositoryPresenter,
                       parametersView: ScanParametersView, plotView: ScanPlotView,
                       fileDialogFactory: FileDialogFactory) -> ScanController:
        controller = cls(repositoryPresenter, parametersView, plotView, fileDialogFactory)
        repositoryPresenter.addObserver(controller)

        controller._proxyModel.setSourceModel(controller._tableModel)

        parametersView.repositoryWidget.tableView.setModel(controller._proxyModel)
        parametersView.repositoryWidget.tableView.setSortingEnabled(True)
        parametersView.repositoryWidget.tableView.setSelectionBehavior(
            QAbstractItemView.SelectRows)
        parametersView.repositoryWidget.tableView.setSelectionMode(
            QAbstractItemView.SingleSelection)
        parametersView.repositoryWidget.tableView.selectionModel().selectionChanged.connect(
            lambda selected, deselected: controller._setButtonsEnabled())
        parametersView.repositoryWidget.tableView.horizontalHeader().sectionClicked.connect(
            lambda logicalIndex: controller._redrawPlot())

        initializerNameList = repositoryPresenter.getInitializerNameList()
        initializerNameList.insert(0, ScanController.OPEN_FILE)

        for name in initializerNameList:
            insertAction = parametersView.repositoryWidget.buttonBox.insertMenu.addAction(name)
            insertAction.triggered.connect(controller._createItemLambda(name))

        parametersView.repositoryWidget.buttonBox.editButton.clicked.connect(
            controller._editSelectedScan)
        parametersView.repositoryWidget.buttonBox.saveButton.clicked.connect(
            controller._saveSelectedScan)
        parametersView.repositoryWidget.buttonBox.removeButton.clicked.connect(
            controller._removeSelectedScan)

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
            self._parametersView.repositoryWidget,
            'Open Scan',
            nameFilters=self._repositoryPresenter.getOpenFileFilterList(),
            selectedNameFilter=self._repositoryPresenter.getOpenFileFilter())

        if filePath:
            self._repositoryPresenter.openScan(filePath, nameFilter)

    def _saveSelectedScan(self) -> None:
        current = self._parametersView.repositoryWidget.tableView.currentIndex()

        if current.isValid():
            filePath, nameFilter = self._fileDialogFactory.getSaveFilePath(
                self._parametersView.repositoryWidget,
                'Save Scan',
                nameFilters=self._repositoryPresenter.getSaveFileFilterList(),
                selectedNameFilter=self._repositoryPresenter.getSaveFileFilter())

            if filePath:
                name = current.sibling(current.row(), 0).data()
                self._repositoryPresenter.saveScan(name, filePath, nameFilter)
        else:
            logger.error('No items are selected!')

    def _getSelectedItemPresenter(self) -> Optional[ScanRepositoryItemPresenter]:
        itemPresenter: Optional[ScanRepositoryItemPresenter] = None
        proxyIndex = self._parametersView.repositoryWidget.tableView.currentIndex()

        if proxyIndex.isValid():
            index = self._proxyModel.mapToSource(proxyIndex)
            itemPresenter = self._repositoryPresenter[index.row()]

        return itemPresenter

    def _editSelectedScan(self) -> None:
        itemPresenter = self._getSelectedItemPresenter()

        if itemPresenter is None:
            logger.error('No items are selected!')
        else:
            item = itemPresenter._item

            if isinstance(item._item, CartesianScanRepositoryItem):
                cartesianDialog = ScanEditorDialog.createInstance(
                    CartesianScanView.createInstance(), self._parametersView)
                cartesianDialog.setWindowTitle(itemPresenter.name)
                cartesianController = CartesianScanController.createInstance(
                    item._item, cartesianDialog.editorView)
                cartesianTransformController = ScanTransformController.createInstance(
                    item, cartesianDialog.transformView)
                cartesianDialog.open()
            elif isinstance(item._item, SpiralScanRepositoryItem):
                spiralDialog = ScanEditorDialog.createInstance(SpiralScanView.createInstance(),
                                                               self._parametersView)
                spiralDialog.setWindowTitle(itemPresenter.name)
                spiralController = SpiralScanController.createInstance(
                    item._item, spiralDialog.editorView)
                spiralTransformController = ScanTransformController.createInstance(
                    item, spiralDialog.transformView)
                spiralDialog.open()
            elif isinstance(item._item, LissajousScanRepositoryItem):
                lissajousDialog = ScanEditorDialog.createInstance(
                    LissajousScanView.createInstance(), self._parametersView)
                lissajousDialog.setWindowTitle(itemPresenter.name)
                lissajousController = LissajousScanController.createInstance(
                    item._item, lissajousDialog.editorView)
                lissajousTransformController = ScanTransformController.createInstance(
                    item, lissajousDialog.transformView)
                lissajousDialog.open()
            elif isinstance(item._item, TabularScanRepositoryItem):
                tabularDialog = ScanEditorDialog.createInstance(TabularScanView.createInstance(),
                                                                self._parametersView)
                tabularDialog.setWindowTitle(itemPresenter.name)
                tabularTransformController = ScanTransformController.createInstance(
                    item, tabularDialog.transformView)
                tabularDialog.open()
            else:
                logger.error('Unknown scan repository item!')

    def _removeSelectedScan(self) -> None:
        current = self._parametersView.repositoryWidget.tableView.currentIndex()

        if current.isValid():
            name = current.sibling(current.row(), 0).data()
            self._repositoryPresenter.removeScan(name)
        else:
            logger.error('No items are selected!')

    def _redrawPlot(self) -> None:
        itemDict = {item.name: item for item in self._repositoryPresenter}
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
        selectionModel = self._parametersView.repositoryWidget.tableView.selectionModel()
        enable = False
        enableRemove = False

        for index in selectionModel.selectedIndexes():
            if index.isValid():
                enable = True
                name = index.sibling(index.row(), 0).data()
                enableRemove |= self._repositoryPresenter.canRemoveScan(name)

        self._parametersView.repositoryWidget.buttonBox.saveButton.setEnabled(enable)
        self._parametersView.repositoryWidget.buttonBox.editButton.setEnabled(enable)
        self._parametersView.repositoryWidget.buttonBox.removeButton.setEnabled(enableRemove)

    def _syncModelToView(self) -> None:
        for item in self._repositoryPresenter:
            item.addObserver(self)

        self._tableModel.beginResetModel()
        self._tableModel.endResetModel()

        self._setButtonsEnabled()
        self._redrawPlot()

    def update(self, observable: Observable) -> None:
        if observable is self._repositoryPresenter:
            self._syncModelToView()
        else:
            for item in self._repositoryPresenter:
                if observable is item and self._tableModel.isChecked(item.name):
                    self._redrawPlot()
                    break

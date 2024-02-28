from __future__ import annotations
import logging

from PyQt5.QtCore import QModelIndex, QSortFilterProxyModel, QStringListModel
from PyQt5.QtWidgets import QAbstractItemView, QDialog

from ...api.observer import SequenceObserver
from ...model.product import ScanRepository
from ...model.scan import ScanRepositoryItem
from ...view.repository import RepositoryTableView
from ...view.scan import ScanPlotView
from ...view.widgets import ComboBoxItemDelegate, ExceptionDialog
from ..data import FileDialogFactory
from .editorFactory import ScanEditorViewControllerFactory
from .tableModel import ScanTableModel

logger = logging.getLogger(__name__)


class ScanController(SequenceObserver[ScanRepositoryItem]):

    def __init__(self, repository: ScanRepository, view: RepositoryTableView,
                 plotView: ScanPlotView, fileDialogFactory: FileDialogFactory,
                 tableModel: ScanTableModel, proxyModel: QSortFilterProxyModel) -> None:
        super().__init__()
        self._repository = repository
        self._view = view
        self._plotView = plotView
        self._fileDialogFactory = fileDialogFactory
        self._tableModel = tableModel
        self._proxyModel = proxyModel
        self._editorFactory = ScanEditorViewControllerFactory()

    @classmethod
    def createInstance(cls, repository: ScanRepository, view: RepositoryTableView,
                       plotView: ScanPlotView,
                       fileDialogFactory: FileDialogFactory) -> ScanController:
        tableModel = ScanTableModel(repository)
        proxyModel = QSortFilterProxyModel()
        proxyModel.setSourceModel(tableModel)
        controller = cls(repository, view, plotView, fileDialogFactory, tableModel, proxyModel)
        proxyModel.dataChanged.connect(
            lambda topLeft, bottomRight, roles: controller._redrawPlot())
        repository.addObserver(controller)

        builderListModel = QStringListModel()
        builderListModel.setStringList([name for name in repository.builderNames()])
        builderItemDelegate = ComboBoxItemDelegate(builderListModel, view.tableView)

        view.tableView.setModel(proxyModel)
        view.tableView.setSortingEnabled(True)
        view.tableView.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        view.tableView.setItemDelegateForColumn(2, builderItemDelegate)
        view.tableView.selectionModel().currentChanged.connect(controller._updateView)
        controller._updateView(QModelIndex(), QModelIndex())

        view.tableView.horizontalHeader().sectionClicked.connect(
            lambda logicalIndex: controller._redrawPlot())

        loadFromFileAction = view.buttonBox.loadMenu.addAction('Open File...')
        loadFromFileAction.triggered.connect(controller._loadCurrentScanFromFile)

        copyAction = view.buttonBox.loadMenu.addAction('Copy...')
        copyAction.triggered.connect(controller._copyToCurrentScan)

        view.copierDialog.setWindowTitle('Copy Scan')
        view.copierDialog.sourceComboBox.setModel(tableModel)
        view.copierDialog.destinationComboBox.setModel(tableModel)
        view.copierDialog.finished.connect(controller._finishCopyingScan)

        view.buttonBox.editButton.clicked.connect(controller._editCurrentScan)
        view.buttonBox.saveButton.clicked.connect(controller._saveCurrentScan)

        return controller

    def _getCurrentItemIndex(self) -> int:
        proxyIndex = self._view.tableView.currentIndex()

        if proxyIndex.isValid():
            modelIndex = self._proxyModel.mapToSource(proxyIndex)
            return modelIndex.row()

        logger.warning('No items are selected!')
        return -1

    def _loadCurrentScanFromFile(self) -> None:
        itemIndex = self._getCurrentItemIndex()

        if itemIndex < 0:
            return

        filePath, nameFilter = self._fileDialogFactory.getOpenFilePath(
            self._view,
            'Open Scan',
            nameFilters=self._repository.getOpenFileFilterList(),
            selectedNameFilter=self._repository.getOpenFileFilter())

        if filePath:
            try:
                self._repository.openScan(itemIndex, filePath, nameFilter)
            except Exception as err:
                logger.exception(err)
                ExceptionDialog.showException('File Reader', err)

    def _copyToCurrentScan(self) -> None:
        itemIndex = self._getCurrentItemIndex()

        if itemIndex >= 0:
            self._view.copierDialog.destinationComboBox.setCurrentIndex(itemIndex)
            self._view.copierDialog.open()

    def _finishCopyingScan(self, result: int) -> None:
        if result == QDialog.DialogCode.Accepted:
            sourceIndex = self._view.copierDialog.sourceComboBox.currentIndex()
            destinationIndex = self._view.copierDialog.destinationComboBox.currentIndex()
            self._repository.copyScan(sourceIndex, destinationIndex)

    def _editCurrentScan(self) -> None:
        itemIndex = self._getCurrentItemIndex()

        if itemIndex < 0:
            return

        itemName = self._repository.getName(itemIndex)
        item = self._repository[itemIndex]
        dialog = self._editorFactory.createEditorDialog(itemName, item, self._view)
        dialog.open()

    def _saveCurrentScan(self) -> None:
        itemIndex = self._getCurrentItemIndex()

        if itemIndex < 0:
            return

        filePath, nameFilter = self._fileDialogFactory.getSaveFilePath(
            self._view,
            'Save Scan',
            nameFilters=self._repository.getSaveFileFilterList(),
            selectedNameFilter=self._repository.getSaveFileFilter())

        if filePath:
            try:
                self._repository.saveScan(itemIndex, filePath, nameFilter)
            except Exception as err:
                logger.exception(err)
                ExceptionDialog.showException('File Writer', err)

    def _redrawPlot(self) -> None:
        self._plotView.axes.clear()

        for row in range(self._proxyModel.rowCount()):
            proxyIndex = self._proxyModel.index(row, 0)
            itemIndex = self._proxyModel.mapToSource(proxyIndex).row()

            if self._tableModel.isItemChecked(itemIndex):
                itemName = self._repository.getName(itemIndex)
                scan = self._repository[itemIndex].getScan()
                x = [point.positionXInMeters for point in scan]
                y = [point.positionYInMeters for point in scan]
                self._plotView.axes.plot(x, y, '.-', label=itemName, linewidth=1.5)

        self._plotView.axes.invert_yaxis()
        self._plotView.axes.axis('equal')
        self._plotView.axes.grid(True)
        self._plotView.axes.set_xlabel('X [m]')
        self._plotView.axes.set_ylabel('Y [m]')

        if len(self._plotView.axes.lines) > 0:
            self._plotView.axes.legend(loc='best')

        self._plotView.figureCanvas.draw()

    def _updateView(self, current: QModelIndex, previous: QModelIndex) -> None:
        enabled = current.isValid()
        self._view.buttonBox.loadButton.setEnabled(enabled)
        self._view.buttonBox.saveButton.setEnabled(enabled)
        self._view.buttonBox.editButton.setEnabled(enabled)
        self._view.buttonBox.analyzeButton.setEnabled(enabled)
        self._redrawPlot()

    def handleItemInserted(self, index: int, item: ScanRepositoryItem) -> None:
        self._tableModel.insertItem(index, item)

    def handleItemChanged(self, index: int, item: ScanRepositoryItem) -> None:
        self._tableModel.updateItem(index, item)

        if self._tableModel.isItemChecked(index):
            self._redrawPlot()

    def handleItemRemoved(self, index: int, item: ScanRepositoryItem) -> None:
        self._tableModel.removeItem(index, item)

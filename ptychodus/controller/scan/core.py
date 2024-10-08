from __future__ import annotations
import logging

from PyQt5.QtCore import QModelIndex, QSortFilterProxyModel, QStringListModel
from PyQt5.QtWidgets import QAbstractItemView, QDialog

from ptychodus.api.observer import SequenceObserver

from ...model.product import ScanAPI, ScanRepository
from ...model.product.scan import ScanRepositoryItem
from ...view.repository import RepositoryTableView
from ...view.scan import ScanPlotView
from ...view.widgets import ComboBoxItemDelegate, ExceptionDialog
from ..data import FileDialogFactory
from .editorFactory import ScanEditorViewControllerFactory
from .tableModel import ScanTableModel

logger = logging.getLogger(__name__)


class ScanController(SequenceObserver[ScanRepositoryItem]):
    def __init__(
        self,
        repository: ScanRepository,
        api: ScanAPI,
        view: RepositoryTableView,
        plotView: ScanPlotView,
        fileDialogFactory: FileDialogFactory,
        tableModel: ScanTableModel,
        tableProxyModel: QSortFilterProxyModel,
    ) -> None:
        super().__init__()
        self._repository = repository
        self._api = api
        self._view = view
        self._plotView = plotView
        self._fileDialogFactory = fileDialogFactory
        self._tableModel = tableModel
        self._tableProxyModel = tableProxyModel
        self._editorFactory = ScanEditorViewControllerFactory()

    @classmethod
    def createInstance(
        cls,
        repository: ScanRepository,
        api: ScanAPI,
        view: RepositoryTableView,
        plotView: ScanPlotView,
        fileDialogFactory: FileDialogFactory,
    ) -> ScanController:
        tableModel = ScanTableModel(repository, api)
        tableProxyModel = QSortFilterProxyModel()
        tableProxyModel.setSourceModel(tableModel)
        controller = cls(
            repository, api, view, plotView, fileDialogFactory, tableModel, tableProxyModel
        )
        tableProxyModel.dataChanged.connect(
            lambda topLeft, bottomRight, roles: controller._redrawPlot()
        )
        repository.addObserver(controller)

        builderListModel = QStringListModel()
        builderListModel.setStringList([name for name in api.builderNames()])
        builderItemDelegate = ComboBoxItemDelegate(builderListModel, view.tableView)

        view.tableView.setModel(tableProxyModel)
        view.tableView.setSortingEnabled(True)
        view.tableView.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        view.tableView.setItemDelegateForColumn(2, builderItemDelegate)
        view.tableView.selectionModel().currentChanged.connect(controller._updateView)
        controller._updateView(QModelIndex(), QModelIndex())

        view.tableView.horizontalHeader().sectionClicked.connect(
            lambda logicalIndex: controller._redrawPlot()
        )

        loadFromFileAction = view.buttonBox.loadMenu.addAction('Open File...')
        loadFromFileAction.triggered.connect(controller._loadCurrentScanFromFile)

        copyAction = view.buttonBox.loadMenu.addAction('Copy...')
        copyAction.triggered.connect(controller._copyToCurrentScan)

        saveToFileAction = view.buttonBox.saveMenu.addAction('Save File...')
        saveToFileAction.triggered.connect(controller._saveCurrentScanToFile)

        syncToSettingsAction = view.buttonBox.saveMenu.addAction('Sync To Settings')
        syncToSettingsAction.triggered.connect(controller._syncCurrentScanToSettings)

        view.copierDialog.setWindowTitle('Copy Scan')
        view.copierDialog.sourceComboBox.setModel(tableModel)
        view.copierDialog.destinationComboBox.setModel(tableModel)
        view.copierDialog.finished.connect(controller._finishCopyingScan)

        view.buttonBox.editButton.clicked.connect(controller._editCurrentScan)

        return controller

    def _getCurrentItemIndex(self) -> int:
        proxyIndex = self._view.tableView.currentIndex()

        if proxyIndex.isValid():
            modelIndex = self._tableProxyModel.mapToSource(proxyIndex)
            return modelIndex.row()

        logger.warning('No current index!')
        return -1

    def _loadCurrentScanFromFile(self) -> None:
        itemIndex = self._getCurrentItemIndex()

        if itemIndex < 0:
            return

        filePath, nameFilter = self._fileDialogFactory.getOpenFilePath(
            self._view,
            'Open Scan',
            nameFilters=self._api.getOpenFileFilterList(),
            selectedNameFilter=self._api.getOpenFileFilter(),
        )

        if filePath:
            try:
                self._api.openScan(itemIndex, filePath, fileType=nameFilter)
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
            self._api.copyScan(sourceIndex, destinationIndex)

    def _editCurrentScan(self) -> None:
        itemIndex = self._getCurrentItemIndex()

        if itemIndex < 0:
            return

        itemName = self._repository.getName(itemIndex)
        item = self._repository[itemIndex]
        dialog = self._editorFactory.createEditorDialog(itemName, item, self._view)
        dialog.open()

    def _saveCurrentScanToFile(self) -> None:
        itemIndex = self._getCurrentItemIndex()

        if itemIndex < 0:
            return

        filePath, nameFilter = self._fileDialogFactory.getSaveFilePath(
            self._view,
            'Save Scan',
            nameFilters=self._api.getSaveFileFilterList(),
            selectedNameFilter=self._api.getSaveFileFilter(),
        )

        if filePath:
            try:
                self._api.saveScan(itemIndex, filePath, nameFilter)
            except Exception as err:
                logger.exception(err)
                ExceptionDialog.showException('File Writer', err)

    def _syncCurrentScanToSettings(self) -> None:
        itemIndex = self._getCurrentItemIndex()

        if itemIndex < 0:
            logger.warning('No current item!')
        else:
            item = self._repository[itemIndex]
            item.syncToSettings()

    def _redrawPlot(self) -> None:
        self._plotView.axes.clear()

        for row in range(self._tableProxyModel.rowCount()):
            proxyIndex = self._tableProxyModel.index(row, 0)
            itemIndex = self._tableProxyModel.mapToSource(proxyIndex).row()

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

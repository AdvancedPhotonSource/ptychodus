from __future__ import annotations
import logging
import sys

from PyQt5.QtCore import (Qt, QAbstractTableModel, QModelIndex, QObject, QSortFilterProxyModel,
                          QVariant)
from PyQt5.QtWidgets import QAbstractItemView

from ...api.visualize import Plot2D
from ...model.metadata import MetadataRepositoryItem
from ...model.object import ObjectRepositoryItem
from ...model.probe import ProbeRepositoryItem
from ...model.product import ProductRepository, ProductRepositoryItem, ProductRepositoryObserver
from ...model.scan import ScanRepositoryItem
from ...view.product import ProductView
from ...view.widgets import ExceptionDialog
from ..data import FileDialogFactory
from .editor import ProductEditorViewController

logger = logging.getLogger(__name__)


class ProductRepositoryTableModel(QAbstractTableModel):

    def __init__(self, repository: ProductRepository, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._repository = repository
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
            try:
                item = self._repository[index.row()]
            except IndexError as err:
                logger.exception(err)
                return value

            metadata = item.getMetadata()
            geometry = item.getGeometry()

            if role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.EditRole:
                if index.column() == 0:
                    value = QVariant(metadata.name.getValue())
                elif index.column() == 1:
                    value = QVariant(
                        f'{metadata.probeEnergyInElectronVolts.getValue() / 1000.:.1f}')
                elif index.column() == 2:
                    value = QVariant(f'{metadata.detectorDistanceInMeters.getValue():.3g}')
                elif index.column() == 3:
                    value = QVariant(f'{geometry.objectPlanePixelWidthInMeters:.3g}')
                elif index.column() == 4:
                    value = QVariant(f'{geometry.objectPlanePixelHeightInMeters:.3g}')
                elif index.column() == 5:
                    value = QVariant(f'{sys.getsizeof(item) / (1024 * 1024):.2f}')

        return value

    def setData(self,
                index: QModelIndex,
                value: QVariant,
                role: int = Qt.ItemDataRole.EditRole) -> bool:
        if index.isValid() and role == Qt.ItemDataRole.EditRole:
            try:
                item = self._repository[index.row()]
            except IndexError as err:
                logger.exception(err)
                return False

            metadata = item.getMetadata()

            if index.column() == 0:
                metadata.name.setValue(value.value())
                return True
            elif index.column() == 1:
                metadata.probeEnergyInElectronVolts.setValue(value.value() * 1000)
                return True
            elif index.column() == 2:
                metadata.detectorDistanceInMeters.setValue(value.value())
                return True

        return False

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._repository)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._header)


class ProductController(ProductRepositoryObserver):

    def __init__(self, repository: ProductRepository, view: ProductView,
                 fileDialogFactory: FileDialogFactory, tableModel: ProductRepositoryTableModel,
                 tableProxyModel: QSortFilterProxyModel) -> None:
        super().__init__()
        self._repository = repository
        self._view = view
        self._fileDialogFactory = fileDialogFactory
        self._tableModel = tableModel
        self._tableProxyModel = tableProxyModel

    @classmethod
    def createInstance(cls, repository: ProductRepository, view: ProductView,
                       fileDialogFactory: FileDialogFactory) -> ProductController:
        tableModel = ProductRepositoryTableModel(repository)
        tableProxyModel = QSortFilterProxyModel()
        tableProxyModel.setSourceModel(tableModel)

        controller = cls(repository, view, fileDialogFactory, tableModel, tableProxyModel)
        repository.addObserver(controller)
        controller._updateInfoText()

        view.tableView.setModel(tableProxyModel)
        view.tableView.setSortingEnabled(True)
        view.tableView.sortByColumn(0, Qt.SortOrder.AscendingOrder)
        view.tableView.verticalHeader().hide()
        view.tableView.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        view.tableView.selectionModel().currentChanged.connect(controller._updateEnabledButtons)
        controller._updateEnabledButtons(QModelIndex(), QModelIndex())

        openFileAction = view.buttonBox.insertMenu.addAction('Open File...')
        openFileAction.triggered.connect(controller._openProduct)

        createNewAction = view.buttonBox.insertMenu.addAction('Create New')
        createNewAction.triggered.connect(controller._createNewProduct)

        view.buttonBox.editButton.clicked.connect(controller._editCurrentProduct)
        view.buttonBox.saveButton.clicked.connect(controller._saveCurrentProduct)
        view.buttonBox.removeButton.clicked.connect(controller._removeCurrentProduct)

        return controller

    def _openProduct(self) -> None:
        filePath, nameFilter = self._fileDialogFactory.getOpenFilePath(
            self._view,
            'Open Product',
            nameFilters=self._repository.getOpenFileFilterList(),
            selectedNameFilter=self._repository.getOpenFileFilter())

        if filePath:
            self._repository.openProduct(filePath, nameFilter)

    def _createNewProduct(self) -> None:
        self._repository.createNewProduct()

    def _saveCurrentProduct(self) -> None:
        current = self._tableProxyModel.mapToSource(self._view.tableView.currentIndex())

        if current.isValid():
            filePath, nameFilter = self._fileDialogFactory.getSaveFilePath(
                self._view,
                'Save Product',
                nameFilters=self._repository.getSaveFileFilterList(),
                selectedNameFilter=self._repository.getSaveFileFilter())

            if filePath:
                try:
                    self._repository.saveProduct(current.row(), filePath, nameFilter)
                except Exception as err:
                    logger.exception(err)
                    ExceptionDialog.showException('File Writer', err)
        else:
            logger.error('No current item!')

    def _editCurrentProduct(self) -> None:
        current = self._tableProxyModel.mapToSource(self._view.tableView.currentIndex())

        if current.isValid():
            product = self._repository[current.row()]
            ProductEditorViewController.editProduct(product, self._view)
        else:
            logger.error('No current item!')

    def _removeCurrentProduct(self) -> None:
        current = self._tableProxyModel.mapToSource(self._view.tableView.currentIndex())

        if current.isValid():
            self._repository.removeProduct(current.row())
        else:
            logger.error('No current item!')

    def _updateEnabledButtons(self, current: QModelIndex, previous: QModelIndex) -> None:
        enabled = current.isValid()
        self._view.buttonBox.saveButton.setEnabled(enabled)
        self._view.buttonBox.editButton.setEnabled(enabled)
        self._view.buttonBox.removeButton.setEnabled(enabled)

    def _updateInfoText(self) -> None:
        infoText = self._repository.getInfoText()
        self._view.infoLabel.setText(infoText)

    def handleItemInserted(self, index: int, item: ProductRepositoryItem) -> None:
        parent = QModelIndex()
        self._tableModel.beginInsertRows(parent, index, index)
        self._tableModel.endInsertRows()
        self._updateInfoText()

    def handleMetadataChanged(self, index: int, item: MetadataRepositoryItem) -> None:
        topLeft = self._tableModel.index(index, 0)
        bottomRight = self._tableModel.index(index, self._tableModel.columnCount() - 1)
        self._tableModel.dataChanged.emit(topLeft, bottomRight, [Qt.ItemDataRole.DisplayRole])
        self._updateInfoText()

    def handleScanChanged(self, index: int, item: ScanRepositoryItem) -> None:
        self._updateInfoText()

    def handleProbeChanged(self, index: int, item: ProbeRepositoryItem) -> None:
        self._updateInfoText()

    def handleObjectChanged(self, index: int, item: ObjectRepositoryItem) -> None:
        self._updateInfoText()

    def handleCostsChanged(self, index: int, costs: Plot2D) -> None:
        self._updateInfoText()

    def handleItemRemoved(self, index: int, item: ProductRepositoryItem) -> None:
        parent = QModelIndex()
        self._tableModel.beginRemoveRows(parent, index, index)
        self._tableModel.endRemoveRows()
        self._updateInfoText()

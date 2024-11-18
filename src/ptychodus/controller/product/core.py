from __future__ import annotations
from collections.abc import Sequence
from typing import Any
import logging

from PyQt5.QtCore import (
    Qt,
    QAbstractTableModel,
    QModelIndex,
    QObject,
    QSortFilterProxyModel,
)
from PyQt5.QtWidgets import QAbstractItemView, QAction

from ...model.product import (
    ProductAPI,
    ProductRepository,
    ProductRepositoryItem,
    ProductRepositoryObserver,
)
from ...model.product.metadata import MetadataRepositoryItem
from ...model.product.object import ObjectRepositoryItem
from ...model.product.probe import ProbeRepositoryItem
from ...model.product.scan import ScanRepositoryItem
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
            'Detector-Object\nDistance [m]',
            'Probe Energy\n[keV]',
            'Probe Photon\nCount',
            'Exposure\nTime [s]',
            'Pixel Width\n[nm]',
            'Pixel Height\n[nm]',
            'Size\n[MB]',
        ]

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        value = super().flags(index)

        if index.isValid() and index.column() < 5:
            value |= Qt.ItemFlag.ItemIsEditable

        return value

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return self._header[section]

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if index.isValid():
            try:
                item = self._repository[index.row()]
            except IndexError as err:
                logger.exception(err)
                return None

            metadata = item.getMetadata()
            geometry = item.getGeometry()

            if role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.EditRole:
                if index.column() == 0:
                    return metadata.getName()
                elif index.column() == 1:
                    return f'{metadata.detectorDistanceInMeters.getValue():.4g}'
                elif index.column() == 2:
                    return f'{metadata.probeEnergyInElectronVolts.getValue() / 1e3:.4g}'
                elif index.column() == 3:
                    return f'{metadata.probePhotonCount.getValue():.4g}'
                elif index.column() == 4:
                    return f'{metadata.exposureTimeInSeconds.getValue():.4g}'
                elif index.column() == 5:
                    return f'{geometry.objectPlanePixelWidthInMeters * 1e9:.4g}'
                elif index.column() == 6:
                    return f'{geometry.objectPlanePixelHeightInMeters * 1e9:.4g}'
                elif index.column() == 7:
                    product = item.getProduct()
                    return f'{product.sizeInBytes / (1024 * 1024):.2f}'

    def setData(self, index: QModelIndex, value: Any, role: int = Qt.ItemDataRole.EditRole) -> bool:
        if index.isValid() and role == Qt.ItemDataRole.EditRole:
            try:
                item = self._repository[index.row()]
            except IndexError as err:
                logger.exception(err)
                return False

            metadata = item.getMetadata()

            if index.column() == 0:
                metadata.setName(str(value))
                return True
            elif index.column() == 1:
                try:
                    distanceInM = float(value)
                except ValueError:
                    return False

                metadata.detectorDistanceInMeters.setValue(distanceInM)
                return True
            elif index.column() == 2:
                try:
                    energyInKEV = float(value)
                except ValueError:
                    return False

                metadata.probeEnergyInElectronVolts.setValue(energyInKEV * 1e3)
                return True
            elif index.column() == 3:
                try:
                    photonCount = float(value)
                except ValueError:
                    return False

                metadata.probePhotonCount.setValue(photonCount)
                return True
            elif index.column() == 4:
                try:
                    exposureTimeInSeconds = float(value)
                except ValueError:
                    return False

                metadata.exposureTimeInSeconds.setValue(exposureTimeInSeconds)
                return True

        return False

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._repository)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._header)


class ProductController(ProductRepositoryObserver):
    def __init__(
        self,
        repository: ProductRepository,
        api: ProductAPI,
        view: ProductView,
        fileDialogFactory: FileDialogFactory,
        duplicateAction: QAction,
        tableModel: ProductRepositoryTableModel,
        tableProxyModel: QSortFilterProxyModel,
    ) -> None:
        super().__init__()
        self._repository = repository
        self._api = api
        self._view = view
        self._fileDialogFactory = fileDialogFactory
        self._duplicateAction = duplicateAction
        self._tableModel = tableModel
        self._tableProxyModel = tableProxyModel

    @classmethod
    def createInstance(
        cls,
        repository: ProductRepository,
        api: ProductAPI,
        view: ProductView,
        fileDialogFactory: FileDialogFactory,
    ) -> ProductController:
        openFileAction = view.buttonBox.insertMenu.addAction('Open File...')
        createNewAction = view.buttonBox.insertMenu.addAction('Create New')
        duplicateAction = view.buttonBox.insertMenu.addAction('Duplicate')
        saveFileAction = view.buttonBox.saveMenu.addAction('Save File...')
        syncToSettingsAction = view.buttonBox.saveMenu.addAction('Sync To Settings')

        tableModel = ProductRepositoryTableModel(repository)
        tableProxyModel = QSortFilterProxyModel()
        tableProxyModel.setSourceModel(tableModel)

        controller = cls(
            repository,
            api,
            view,
            fileDialogFactory,
            duplicateAction,
            tableModel,
            tableProxyModel,
        )
        repository.addObserver(controller)
        controller._updateInfoText()

        view.tableView.setModel(tableProxyModel)
        view.tableView.setSortingEnabled(True)
        view.tableView.sortByColumn(0, Qt.SortOrder.AscendingOrder)
        view.tableView.verticalHeader().hide()
        view.tableView.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        view.tableView.selectionModel().currentChanged.connect(controller._updateEnabledButtons)
        controller._updateEnabledButtons(QModelIndex(), QModelIndex())

        openFileAction.triggered.connect(controller._openProductFromFile)
        createNewAction.triggered.connect(controller._createNewProduct)
        duplicateAction.triggered.connect(controller._duplicateCurrentProduct)
        saveFileAction.triggered.connect(controller._saveCurrentProductToFile)
        syncToSettingsAction.triggered.connect(controller._syncCurrentProductToSettings)

        view.buttonBox.editButton.clicked.connect(controller._editCurrentProduct)
        view.buttonBox.removeButton.clicked.connect(controller._removeCurrentProduct)

        return controller

    @property
    def tableModel(self) -> QAbstractTableModel:
        return self._tableModel

    def _getCurrentItemIndex(self) -> int:
        proxyIndex = self._view.tableView.currentIndex()

        if proxyIndex.isValid():
            modelIndex = self._tableProxyModel.mapToSource(proxyIndex)
            return modelIndex.row()

        logger.warning('No current index!')
        return -1

    def _openProductFromFile(self) -> None:
        filePath, nameFilter = self._fileDialogFactory.getOpenFilePath(
            self._view,
            'Open Product',
            nameFilters=self._api.getOpenFileFilterList(),
            selectedNameFilter=self._api.getOpenFileFilter(),
        )

        if filePath:
            try:
                self._api.openProduct(filePath, fileType=nameFilter)
            except Exception as err:
                logger.exception(err)
                ExceptionDialog.showException('File Reader', err)

    def _createNewProduct(self) -> None:
        self._api.insertNewProduct()

    def _saveCurrentProductToFile(self) -> None:
        current = self._tableProxyModel.mapToSource(self._view.tableView.currentIndex())

        if current.isValid():
            filePath, nameFilter = self._fileDialogFactory.getSaveFilePath(
                self._view,
                'Save Product',
                nameFilters=self._api.getSaveFileFilterList(),
                selectedNameFilter=self._api.getSaveFileFilter(),
            )

            if filePath:
                try:
                    self._api.saveProduct(current.row(), filePath, fileType=nameFilter)
                except Exception as err:
                    logger.exception(err)
                    ExceptionDialog.showException('File Writer', err)
        else:
            logger.error('No current item!')

    def _syncCurrentProductToSettings(self) -> None:
        itemIndex = self._getCurrentItemIndex()

        if itemIndex < 0:
            logger.warning('No current item!')
        else:
            item = self._repository[itemIndex]
            item.syncToSettings()

    def _duplicateCurrentProduct(self) -> None:
        current = self._tableProxyModel.mapToSource(self._view.tableView.currentIndex())

        if current.isValid():
            self._api.insertNewProduct(likeIndex=current.row())
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
        self._duplicateAction.setEnabled(enabled)
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

    def handleCostsChanged(self, index: int, costs: Sequence[float]) -> None:
        self._updateInfoText()

    def handleItemRemoved(self, index: int, item: ProductRepositoryItem) -> None:
        parent = QModelIndex()
        self._tableModel.beginRemoveRows(parent, index, index)
        self._tableModel.endRemoveRows()
        self._updateInfoText()

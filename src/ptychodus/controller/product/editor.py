from typing import Any

from PyQt5.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QObject,
    QSortFilterProxyModel,
    Qt,
)
from PyQt5.QtWidgets import QWidget

from ptychodus.api.observer import Observable, Observer

from ...model.patterns import AssembledDiffractionDataset
from ...model.product import ProductRepositoryItem
from ...view.product import ProductEditorDialog


class ProductPropertyTableModel(QAbstractTableModel):
    def __init__(self, product: ProductRepositoryItem, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._product = product
        self._header = ['Property', 'Value']
        self._properties = [
            'Probe Wavelength [nm]',
            'Probe Wavenumber [1/nm]',
            'Probe Angular Wavenumber [rad/nm]',
            'Probe Photon Flux [ph/s]',
            'Probe Power [W]',
            'Object Plane Pixel Width [nm]',
            'Object Plane Pixel Height [nm]',
            'Fresnel Number',
        ]

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return self._header[section]

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if index.isValid() and role == Qt.ItemDataRole.DisplayRole:
            match index.column():
                case 0:
                    return self._properties[index.row()]
                case 1:
                    geometry = self._product.get_geometry()

                    match index.row():
                        case 0:
                            return f'{geometry.probe_wavelength_m * 1e9:.4g}'
                        case 1:
                            return f'{geometry.probeWavelengthsPerMeter * 1e-9:.4g}'
                        case 2:
                            return f'{geometry.probeRadiansPerMeter * 1e-9:.4g}'
                        case 3:
                            return f'{geometry.probePhotonsPerSecond:.4g}'
                        case 4:
                            return f'{geometry.probe_power_W:.4g}'
                        case 5:
                            return f'{geometry.objectPlanePixelWidthInMeters * 1e9:.4g}'
                        case 6:
                            return f'{geometry.objectPlanePixelHeightInMeters * 1e9:.4g}'
                        case 7:
                            try:
                                return f'{geometry.fresnelNumber:.4g}'
                            except ZeroDivisionError:
                                return 'inf'

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._properties)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._header)


class ProductEditorViewController(Observer):
    def __init__(
        self,
        dataset: AssembledDiffractionDataset,
        product: ProductRepositoryItem,
        tableModel: ProductPropertyTableModel,
        dialog: ProductEditorDialog,
    ) -> None:
        super().__init__()
        self._dataset = dataset
        self._product = product
        self._tableModel = tableModel
        self._dialog = dialog

    @classmethod
    def editProduct(
        cls, dataset: AssembledDiffractionDataset, product: ProductRepositoryItem, parent: QWidget
    ) -> None:
        tableModel = ProductPropertyTableModel(product)
        tableProxyModel = QSortFilterProxyModel()
        tableProxyModel.setSourceModel(tableModel)

        dialog = ProductEditorDialog(parent)
        dialog.setWindowTitle(f'Edit Product: {product.get_name()}')
        dialog.tableView.setModel(tableProxyModel)
        dialog.tableView.setSortingEnabled(True)
        dialog.tableView.verticalHeader().hide()
        dialog.tableView.resizeColumnsToContents()
        dialog.tableView.resizeRowsToContents()

        viewController = cls(dataset, product, tableModel, dialog)
        product.add_observer(viewController)
        dialog.textEdit.textChanged.connect(viewController._syncViewToModel)

        viewController._syncModelToView()

        dialog.actionsView.estimateProbePhotonCountButton.clicked.connect(
            viewController._estimateProbePhotonCount
        )
        dialog.finished.connect(viewController._finish)
        dialog.open()
        dialog.adjustSize()

    def _syncViewToModel(self) -> None:
        metadata = self._product.getMetadata()
        metadata.comments.set_value(self._dialog.textEdit.toPlainText())

    def _syncModelToView(self) -> None:
        self._tableModel.beginResetModel()
        self._tableModel.endResetModel()

        metadata = self._product.getMetadata()
        self._dialog.textEdit.setPlainText(metadata.comments.get_value())

    def _estimateProbePhotonCount(self) -> None:
        metadata = self._product.getMetadata()
        metadata.probePhotonCount.set_value(self._dataset.get_maximum_pattern_counts())

        self._tableModel.beginResetModel()
        self._tableModel.endResetModel()

    def _finish(self, result: int) -> None:
        self._product.remove_observer(self)

    def _update(self, observable: Observable) -> None:
        if observable is self._product:
            self._syncModelToView()

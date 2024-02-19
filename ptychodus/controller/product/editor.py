from PyQt5.QtCore import (Qt, QAbstractTableModel, QModelIndex, QObject, QSortFilterProxyModel,
                          QVariant)
from PyQt5.QtWidgets import QWidget

from ...api.observer import Observable, Observer
from ...model.product import ProductRepositoryItem
from ...view.product import ProductEditorDialog


class ProductPropertyTableModel(QAbstractTableModel):

    def __init__(self, product: ProductRepositoryItem, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._product = product
        self._header = ['Property', 'Value']
        self._properties = [
            'Probe Wavelength [nm]',
            'Fresnel Number',
            'Object Plane Pixel Width [nm]',
            'Object Plane Pixel Height [nm]',
        ]

    def headerData(self,
                   section: int,
                   orientation: Qt.Orientation,
                   role: int = Qt.ItemDataRole.DisplayRole) -> QVariant:
        result = QVariant()

        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            result = QVariant(self._header[section])

        return result

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> QVariant:
        if index.isValid() and role == Qt.ItemDataRole.DisplayRole:
            if index.column() == 0:
                return QVariant(self._properties[index.row()])
            elif index.column() == 1:
                geometry = self._product.getGeometry()

                if index.row() == 0:
                    return QVariant(f'{geometry.probeWavelengthInMeters * 1.e9:.4g}')
                elif index.row() == 1:
                    return QVariant(f'{geometry.fresnelNumber:.4g}')
                elif index.row() == 2:
                    return QVariant(f'{geometry.objectPlanePixelWidthInMeters * 1.e9:.4g}')
                elif index.row() == 3:
                    return QVariant(f'{geometry.objectPlanePixelHeightInMeters * 1.e9:.4g}')

        return QVariant()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._properties)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._header)


class ProductEditorViewController(Observer):

    def __init__(self, product: ProductRepositoryItem,
                 tableModel: ProductPropertyTableModel) -> None:
        super().__init__()
        self._product = product
        self._tableModel = tableModel

    @classmethod
    def editProduct(cls, product: ProductRepositoryItem, parent: QWidget) -> None:
        tableModel = ProductPropertyTableModel(product)
        controller = cls(product, tableModel)
        product.addObserver(controller)

        tableProxyModel = QSortFilterProxyModel()
        tableProxyModel.setSourceModel(tableModel)

        dialog = ProductEditorDialog.createInstance(parent)
        dialog.setWindowTitle(f'Edit Product: {product.getName()}')
        dialog.tableView.setModel(tableProxyModel)
        dialog.tableView.setSortingEnabled(True)
        dialog.tableView.verticalHeader().hide()
        dialog.finished.connect(controller._finish)

        controller._syncModelToView()
        dialog.open()

    def _finish(self, result: int) -> None:
        self._product.removeObserver(self)

    def _syncModelToView(self) -> None:
        self._tableModel.beginResetModel()
        self._tableModel.endResetModel()

    def update(self, observable: Observable) -> None:
        if observable is self._product:
            self._syncModelToView()

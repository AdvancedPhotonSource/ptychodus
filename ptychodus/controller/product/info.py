from PyQt5.QtCore import (Qt, QAbstractTableModel, QModelIndex, QObject, QSortFilterProxyModel,
                          QVariant)
from PyQt5.QtWidgets import QWidget

from ...api.observer import Observable, Observer
from ...model.product import ProductRepositoryItem
from ...view.product import ProductInfoDialog


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
            'Resolution Gain',  # FIXME
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
        value = QVariant()

        if index.isValid() and role == Qt.ItemDataRole.DisplayRole:
            if index.column() == 0:
                value = QVariant(self._properties[index.row()])
            elif index.column() == 1:
                metadata = self._product.getMetadata()

                if index.row() == 0:
                    probeEnergy = metadata.probeEnergyInElectronVolts.getValue()
                    value = QVariant(f'{probeEnergy / 1000:.1f}')
                else:
                    value = QVariant(index.row())  # FIXME

        return value

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._properties)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._header)


class ProductInfoViewController(Observer):

    def __init__(self, product: ProductRepositoryItem,
                 tableModel: ProductPropertyTableModel) -> None:
        super().__init__()
        self._product = product
        self._tableModel = tableModel

    @classmethod
    def showInfo(cls, product: ProductRepositoryItem, parent: QWidget) -> None:
        tableModel = ProductPropertyTableModel(product)
        controller = cls(product, tableModel)
        product.addObserver(controller)

        tableProxyModel = QSortFilterProxyModel()
        tableProxyModel.setSourceModel(tableModel)

        dialog = ProductInfoDialog.createInstance(parent)
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

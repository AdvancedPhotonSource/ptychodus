from typing import Any

from PyQt5.QtCore import (
    Qt,
    QAbstractTableModel,
    QModelIndex,
    QObject,
    QSortFilterProxyModel,
)
from PyQt5.QtWidgets import QWidget

from ptychodus.api.observer import Observable, Observer

from ...model.product import ProductRepositoryItem
from ...view.product import ProductEditorDialog


class ProductPropertyTableModel(QAbstractTableModel):

    def __init__(self, product: ProductRepositoryItem, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._product = product
        self._header = ["Property", "Value"]
        self._properties = [
            "Probe Wavelength [nm]",
            "Probe Power [W]",
            "Object Plane Pixel Width [nm]",
            "Object Plane Pixel Height [nm]",
            "Fresnel Number",
        ]

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if (orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole):
            return self._header[section]

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if index.isValid() and role == Qt.ItemDataRole.DisplayRole:
            if index.column() == 0:
                return self._properties[index.row()]
            elif index.column() == 1:
                geometry = self._product.getGeometry()

                if index.row() == 0:
                    return f"{geometry.probeWavelengthInMeters * 1e9:.4g}"
                elif index.row() == 1:
                    return f"{geometry.probePowerInWatts:.4g}"
                elif index.row() == 2:
                    return f"{geometry.objectPlanePixelWidthInMeters * 1e9:.4g}"
                elif index.row() == 3:
                    return f"{geometry.objectPlanePixelHeightInMeters * 1e9:.4g}"
                elif index.row() == 4:
                    return f"{geometry.fresnelNumber:.4g}"

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._properties)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._header)


class ProductEditorViewController(Observer):

    def __init__(
        self,
        product: ProductRepositoryItem,
        tableModel: ProductPropertyTableModel,
        dialog: ProductEditorDialog,
    ) -> None:
        super().__init__()
        self._product = product
        self._tableModel = tableModel
        self._dialog = dialog

    @classmethod
    def editProduct(cls, product: ProductRepositoryItem, parent: QWidget) -> None:
        tableModel = ProductPropertyTableModel(product)
        tableProxyModel = QSortFilterProxyModel()
        tableProxyModel.setSourceModel(tableModel)

        dialog = ProductEditorDialog.createInstance(parent)
        dialog.setWindowTitle(f"Edit Product: {product.getName()}")
        dialog.tableView.setModel(tableProxyModel)
        dialog.tableView.setSortingEnabled(True)
        dialog.tableView.verticalHeader().hide()
        dialog.tableView.resizeColumnsToContents()
        dialog.tableView.resizeRowsToContents()

        viewController = cls(product, tableModel, dialog)
        product.addObserver(viewController)
        dialog.textEdit.textChanged.connect(viewController._syncViewToModel)

        viewController._syncModelToView()
        dialog.finished.connect(viewController._finish)
        dialog.open()
        dialog.adjustSize()

    def _syncViewToModel(self) -> None:
        metadata = self._product.getMetadata()
        metadata.comments.setValue(self._dialog.textEdit.toPlainText())

    def _syncModelToView(self) -> None:
        self._tableModel.beginResetModel()
        self._tableModel.endResetModel()

        metadata = self._product.getMetadata()
        self._dialog.textEdit.setPlainText(metadata.comments.getValue())

    def _finish(self, result: int) -> None:
        self._product.removeObserver(self)

    def update(self, observable: Observable) -> None:
        if observable is self._product:
            self._syncModelToView()

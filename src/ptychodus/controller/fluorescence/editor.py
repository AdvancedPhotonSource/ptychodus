from __future__ import annotations
from collections.abc import Sequence
from enum import Enum, IntEnum
from typing import Any

from PyQt5.QtCore import (
    QAbstractItemModel,
    QAbstractTableModel,
    QModelIndex,
    QObject,
    Qt,
)
from PyQt5.QtGui import QBrush
from PyQt5.QtWidgets import QComboBox, QStyledItemDelegate, QStyleOptionViewItem, QWidget

from ptychodus.api.constants import format_bytes

from ...model.fluorescence import FluorescenceRepositoryItem
from ptychodus.api.product import LossValue

from ...model.product import (
    ProductRepository,
    ProductRepositoryItem,
    ProductRepositoryObserver,
)
from ...model.product.metadata import MetadataRepositoryItem
from ...model.product.object import ObjectRepositoryItem
from ...model.product.probe import ProbeRepositoryItem
from ...model.product.probe_positions import ProbePositionsRepositoryItem
from ...view.fluorescence import FluorescenceEditorDialog
from ..helpers import create_brush_for_editable_cell


class _Row(Enum):
    """Every row of FluorescencePropertyTableModel: its order, its label and whether it edits.

    The member value pairs the label with editability, and the position comes
    from declaration order rather than being written out, so a row cannot be added
    to the ordering and forgotten in the labels or the editable set. Nothing could pin
    that alignment: CLAUDE.md puts this file outside the tested surface.

    Bare names in a ``match`` arm are capture patterns, not value patterns.
    """

    NAME = ('Name', True)
    PRODUCT = ('Product', True)
    SOURCE_PATH = ('Source Path', False)
    FILE_TYPE = ('File Type', False)
    ELEMENTS = ('Elements', False)
    COUNTS_PER_SECOND_PATH = ('Counts Per Second Path', False)
    CHANNEL_NAMES_PATH = ('Channel Names Path', False)
    SIZE = ('Size', False)

    def __init__(self, label: str, editable: bool) -> None:
        self.label = label
        self.editable = editable
        # Members already defined when this one is constructed, which is its position.
        self.index = len(self.__class__.__members__)

    @classmethod
    def from_index(cls, index: int) -> _Row:
        """The row at `index`.

        Raises:
            IndexError: for a row outside the table, which the model's own
                ``rowCount`` makes impossible and so would be a bug rather than input.
        """
        if index < 0:
            raise IndexError(f'Row index {index} is negative.')

        return list(cls)[index]


class _Col(IntEnum):
    """Column order of FluorescencePropertyTableModel.

    An enum for the same reason as _Row: bare names in a ``match`` arm are capture
    patterns, not value patterns.
    """

    PROPERTY = 0
    VALUE = 1


class FluorescencePropertyTableModel(QAbstractTableModel):
    """Property/value table over one fluorescence repository item."""

    def __init__(
        self,
        item: FluorescenceRepositoryItem,
        product_repository: ProductRepository,
        editable_item_brush: QBrush,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._item = item
        self._product_repository = product_repository
        self._editable_item_brush = editable_item_brush
        self._header = ['Property', 'Value']

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        value = super().flags(index)

        # Only the Value column is editable: the Property column holds the row's
        # read-only label, and setData keys off the row alone.
        if index.isValid() and index.column() == _Col.VALUE:
            if _Row.from_index(index.row()).editable:
                value |= Qt.ItemFlag.ItemIsEditable

        return value

    def headerData(  # noqa: N802
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return self._header[section]

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None

        if role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.EditRole:
            if index.column() == _Col.PROPERTY:
                return _Row.from_index(index.row()).label

            if index.column() == _Col.VALUE:
                return self._value(_Row.from_index(index.row()))
        elif role == Qt.ItemDataRole.BackgroundRole:
            if index.flags() & Qt.ItemFlag.ItemIsEditable:
                return self._editable_item_brush

        return None

    def _value(self, row: _Row) -> Any:
        measured = self._item.get_measured()

        match row:
            case _Row.NAME:
                return self._item.get_name()
            case _Row.PRODUCT:
                return self._item.get_product().get_name()
            case _Row.SOURCE_PATH:
                source_path = self._item.get_source_path()
                return '(none)' if source_path is None else str(source_path)
            case _Row.FILE_TYPE:
                return self._item.get_source_file_type() or '(none)'
            case _Row.ELEMENTS:
                return len(measured)
            case _Row.COUNTS_PER_SECOND_PATH:
                return measured.counts_per_second_path
            case _Row.CHANNEL_NAMES_PATH:
                return measured.channel_names_path
            case _Row.SIZE:
                return format_bytes(self._item.get_nbytes())

    def setData(self, index: QModelIndex, value: Any, role: int = Qt.ItemDataRole.EditRole) -> bool:  # noqa: N802
        if not index.isValid() or role != Qt.ItemDataRole.EditRole:
            return False

        if index.column() != _Col.VALUE:
            return False

        match _Row.from_index(index.row()):
            case _Row.NAME:
                self._item.set_name(str(value))
                return True
            case _Row.PRODUCT:
                for product in self._product_repository:
                    if product.get_name() == str(value):
                        self._item.set_product(product)
                        return True

                return False
            case _:
                return False

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(_Row)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self._header)


class _ProductDelegate(QStyledItemDelegate):
    """Combobox editor for the bound-product row.

    Only one delegate can be installed per column, so this dispatches on the row and
    lets every other row fall through to the base delegate.
    """

    def __init__(self, product_repository: ProductRepository, parent: QWidget) -> None:
        super().__init__(parent)
        self._product_repository = product_repository

    def _is_product_cell(self, index: QModelIndex) -> bool:
        return index.column() == _Col.VALUE and _Row.from_index(index.row()) is _Row.PRODUCT

    def createEditor(  # noqa: N802
        self, parent: QWidget, option: QStyleOptionViewItem, index: QModelIndex
    ) -> QWidget:
        if not self._is_product_cell(index):
            return super().createEditor(parent, option, index)

        combo = QComboBox(parent)

        for product in self._product_repository:
            combo.addItem(product.get_name())

        return combo

    def setEditorData(self, editor: QWidget, index: QModelIndex) -> None:  # noqa: N802
        if not isinstance(editor, QComboBox):
            super().setEditorData(editor, index)
            return

        target = editor.findText(str(index.data(Qt.ItemDataRole.DisplayRole) or ''))
        editor.setCurrentIndex(max(0, target))

    def setModelData(  # noqa: N802
        self, editor: QWidget, model: QAbstractItemModel, index: QModelIndex
    ) -> None:
        if not isinstance(editor, QComboBox):
            super().setModelData(editor, model, index)
            return

        model.setData(index, editor.currentText(), Qt.ItemDataRole.EditRole)


class FluorescenceEditorViewController(ProductRepositoryObserver):
    """Drives FluorescenceEditorDialog for a single repository item.

    Observes the ProductRepository because the Product row renders a product name and
    its editor enumerates the repository: without this, both go stale when a product
    is renamed, added, or removed while the dialog is open. The fluorescence item
    itself is not Observable -- it uses the parent-callback pattern -- so there is
    nothing to register on for changes to the item.
    """

    def __init__(
        self,
        item: FluorescenceRepositoryItem,
        product_repository: ProductRepository,
        table_model: FluorescencePropertyTableModel,
        dialog: FluorescenceEditorDialog,
    ) -> None:
        super().__init__()
        self._item = item
        self._product_repository = product_repository
        self._table_model = table_model
        self._dialog = dialog

    @classmethod
    def edit_item(
        cls,
        product_repository: ProductRepository,
        item: FluorescenceRepositoryItem,
        parent: QWidget,
    ) -> None:
        dialog = FluorescenceEditorDialog(parent)
        dialog.setWindowTitle(f'Edit Fluorescence: {item.get_name()}')

        table_model = FluorescencePropertyTableModel(
            item, product_repository, create_brush_for_editable_cell(dialog.table_view)
        )
        dialog.table_view.setModel(table_model)
        dialog.table_view.setItemDelegateForColumn(
            _Col.VALUE, _ProductDelegate(product_repository, dialog.table_view)
        )

        vertical_header = dialog.table_view.verticalHeader()

        if vertical_header is not None:
            vertical_header.hide()

        header = dialog.table_view.horizontalHeader()
        header.setSectionResizeMode(header.ResizeMode.ResizeToContents)
        dialog.table_view.resizeRowsToContents()

        view_controller = cls(item, product_repository, table_model, dialog)
        product_repository.add_observer(view_controller)
        dialog.finished.connect(view_controller._finish)
        dialog.open()
        dialog.adjustSize()

    def _sync_model_to_view(self) -> None:
        self._table_model.beginResetModel()
        self._table_model.endResetModel()

    # Any change to the product list moves the Product row's text or the combo's
    # contents; everything else on this dialog is item-local.

    def handle_item_inserted(self, index: int, item: ProductRepositoryItem) -> None:
        self._sync_model_to_view()

    def handle_item_removed(self, index: int, item: ProductRepositoryItem) -> None:
        self._sync_model_to_view()

    def handle_metadata_changed(self, index: int, item: MetadataRepositoryItem) -> None:
        self._sync_model_to_view()

    def handle_probe_positions_changed(
        self, index: int, item: ProbePositionsRepositoryItem
    ) -> None:
        pass

    def handle_probe_changed(self, index: int, item: ProbeRepositoryItem) -> None:
        pass

    def handle_object_changed(self, index: int, item: ObjectRepositoryItem) -> None:
        pass

    def handle_losses_changed(self, index: int, losses: Sequence[LossValue]) -> None:
        pass

    def handle_dataset_changed(self, index: int, item: ProductRepositoryItem) -> None:
        pass

    def handle_state_changed(self, index: int, item: ProductRepositoryItem) -> None:
        pass

    def _finish(self, result: int) -> None:
        self._product_repository.remove_observer(self)
        self._dialog.deleteLater()

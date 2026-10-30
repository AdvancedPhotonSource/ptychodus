from __future__ import annotations
from enum import Enum
from typing import Any, overload

import numpy

from PyQt5.QtCore import Qt, QAbstractItemModel, QModelIndex, QObject
from PyQt5.QtGui import QBrush

from ptychodus.api.constants import LengthUnit, format_bytes
from ptychodus.api.probe import Probe

from ...model.product import ProbeAPI, ProbeRepository
from ...model.product.probe import ProbeRepositoryItem


class ProbeTreeNode:
    def __init__(self, parent: ProbeTreeNode | None = None) -> None:
        self.parent = parent
        self.children: list[ProbeTreeNode] = list()

    def insert_node(self, index: int = -1) -> ProbeTreeNode:
        node = ProbeTreeNode(self)
        self.children.insert(index, node)
        return node

    def remove_node(self, index: int = -1) -> ProbeTreeNode:
        return self.children.pop(index)

    def row(self) -> int:
        return 0 if self.parent is None else self.parent.children.index(self)


def calc_relative_power_percent(probe: Probe, imode: int) -> int:
    try:
        relative_power = probe.get_incoherent_mode_relative_power(imode)
    except IndexError:
        return -1

    if numpy.isfinite(relative_power):
        return int(100.0 * relative_power)

    return -1


def calc_coherent_percent(probe: Probe) -> int:
    coherence = probe.get_coherence()
    return int(100.0 * coherence) if numpy.isfinite(coherence) else -1


def try_get_probe(item: ProbeRepositoryItem) -> Probe | None:
    # Returns None when the probe sequence is the null sentinel (no dataset
    # bound yet — see ProbeRepositoryItem._rebuild). Same pattern used by
    # ProbeRepositoryItem.get_size_metrics / get_entropy_metrics.
    try:
        return item.get_probes().get_probe_no_opr()
    except ValueError:
        return None


class Column(Enum):
    """Every column of ProbeTreeModel: its order, its header and whether it edits.

    The member value pairs the header with editability, and the position comes
    from declaration order rather than being written out, so a column cannot be added
    to the ordering and forgotten in the labels or the editable set. Nothing could pin
    that alignment: CLAUDE.md puts this file outside the tested surface.

    Public because the controller installs item delegates on some of them.

    ``editable`` is the item row's sense; a child row is governed by its own arm.
    """

    NAME = ('Name', True)
    RELATIVE_POWER = ('Relative Power', False)
    BUILDER = ('Builder', True)
    DATA_TYPE = ('Data Type', False)
    WIDTH_PX = ('Width [px]', False)
    HEIGHT_PX = ('Height [px]', False)
    PIXEL_WIDTH_NM = ('Pixel Width\n[nm]', False)
    PIXEL_HEIGHT_NM = ('Pixel Height\n[nm]', False)
    SIZE = ('Size', False)

    def __init__(self, header: str, editable: bool) -> None:
        self.header = header
        self.editable = editable
        # Members already defined when this one is constructed, which is its position.
        self.index = len(self.__class__.__members__)

    @classmethod
    def from_index(cls, index: int) -> Column:
        """The column at `index`.

        Raises:
            IndexError: for a column outside the table, which the model's own
                ``columnCount`` makes impossible and so would be a bug rather than input.
        """
        if index < 0:
            raise IndexError(f'Column index {index} is negative.')

        return list(cls)[index]


class ProbeTreeModel(QAbstractItemModel):
    def __init__(
        self,
        repository: ProbeRepository,
        api: ProbeAPI,
        editable_item_brush: QBrush,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._repository = repository
        self._api = api
        self._editable_item_brush = editable_item_brush
        self._tree_root = ProbeTreeNode()

        for index, item in enumerate(repository):
            self.insert_item(index, item)

    @staticmethod
    def _append_modes(node: ProbeTreeNode, item: ProbeRepositoryItem) -> None:
        probe = item.get_probes()

        for layer in range(probe.num_incoherent_modes):
            node.insert_node()

    def insert_item(self, index: int, item: ProbeRepositoryItem) -> None:
        self.beginInsertRows(QModelIndex(), index, index)
        ProbeTreeModel._append_modes(self._tree_root.insert_node(index), item)
        self.endInsertRows()

    def update_item(self, index: int, item: ProbeRepositoryItem) -> None:
        # Qt row/column bounds are inclusive, so the last valid column is
        # len(Column) - 1. Asking index() for one past the end fails
        # hasIndex and yields an invalid QModelIndex, which makes the
        # dataChanged range malformed and the signal a no-op.
        last_column = len(Column) - 1
        top_left = self.index(index, 0)
        bottom_right = self.index(index, last_column)
        self.dataChanged.emit(top_left, bottom_right)

        node = self._tree_root.children[index]
        num_modes_old = len(node.children)
        num_modes_new = item.get_probes().num_incoherent_modes

        if num_modes_old < num_modes_new:
            self.beginInsertRows(top_left, num_modes_old, num_modes_new)

            while len(node.children) < num_modes_new:
                node.insert_node()

            self.endInsertRows()
        elif num_modes_old > num_modes_new:
            self.beginRemoveRows(top_left, num_modes_new, num_modes_old)

            while len(node.children) > num_modes_new:
                node.remove_node()

            self.endRemoveRows()

        # Both corners were one past the end here, in row as well as column.
        if num_modes_new > 0:
            child_top_left = self.index(0, 0, top_left)
            child_bottom_right = self.index(num_modes_new - 1, last_column, top_left)
            self.dataChanged.emit(child_top_left, child_bottom_right)

    def remove_item(self, index: int, item: ProbeRepositoryItem) -> None:
        self.beginRemoveRows(QModelIndex(), index, index)
        self._tree_root.remove_node(index)
        self.endRemoveRows()

    def headerData(  # noqa: N802
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return Column.from_index(section).header

    @overload
    def parent(self, child: QModelIndex) -> QModelIndex: ...

    @overload
    def parent(self) -> QObject: ...

    def parent(self, child: QModelIndex | None = None) -> QModelIndex | QObject:
        if child is None:
            return super().parent()
        elif child.isValid():
            node = child.internalPointer()
            parent_node = node.parent
            return (
                QModelIndex()
                if parent_node is self._tree_root
                else self.createIndex(parent_node.row(), 0, parent_node)
            )

        return QModelIndex()

    def index(self, row: int, column: int, parent: QModelIndex = QModelIndex()) -> QModelIndex:
        if self.hasIndex(row, column, parent):
            if parent.isValid():
                parent_node = parent.internalPointer()
                node = parent_node.children[row]
            else:
                node = self._tree_root.children[row]

            return self.createIndex(row, column, node)

        return QModelIndex()

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if not index.isValid():
            return None

        parent = index.parent()

        if parent.isValid():
            item = self._repository[parent.row()]

            if role == Qt.ItemDataRole.DisplayRole:
                match Column.from_index(index.column()):
                    case Column.NAME:
                        return f'Mode {index.row() + 1}'
                    case Column.RELATIVE_POWER:
                        probe = try_get_probe(item)
                        if probe is None:
                            return None
                        power_percent = calc_relative_power_percent(probe, index.row())
                        return f'{power_percent}%'
            elif role == Qt.ItemDataRole.BackgroundRole:
                if index.flags() & Qt.ItemFlag.ItemIsEditable:
                    return self._editable_item_brush
            elif (
                role == Qt.ItemDataRole.UserRole
                and Column.from_index(index.column()) is Column.RELATIVE_POWER
            ):
                probe = try_get_probe(item)
                if probe is None:
                    return None
                return calc_relative_power_percent(probe, index.row())
        else:
            item = self._repository[index.row()]
            probes = item.get_probes()
            probe = try_get_probe(item)
            # None when the probe is not yet built (see ProbeRepositoryItem._rebuild
            # guard): probe-dependent columns return None; name/builder/size still show.
            pixel_geometry = probe.get_pixel_geometry() if probe is not None else None

            if role == Qt.ItemDataRole.DisplayRole:
                match Column.from_index(index.column()):
                    case Column.NAME:
                        return self._repository.get_name(index.row())
                    case Column.RELATIVE_POWER:
                        if probe is None:
                            return None
                        coherent_percent = calc_coherent_percent(probe)
                        return f'{coherent_percent}%'
                    case Column.BUILDER:
                        return item.get_builder().get_name()
                    case Column.DATA_TYPE:
                        return str(probe.dtype) if probe is not None else None
                    case Column.WIDTH_PX:
                        return probe.width_px if probe is not None else None
                    case Column.HEIGHT_PX:
                        return probe.height_px if probe is not None else None
                    case Column.PIXEL_WIDTH_NM:
                        return (
                            f'{LengthUnit.NANOMETER.convert(pixel_geometry.width_m):.4g}'
                            if pixel_geometry
                            else None
                        )
                    case Column.PIXEL_HEIGHT_NM:
                        return (
                            f'{LengthUnit.NANOMETER.convert(pixel_geometry.height_m):.4g}'
                            if pixel_geometry
                            else None
                        )
                    case Column.SIZE:
                        return format_bytes(probes.nbytes)
            elif role == Qt.ItemDataRole.BackgroundRole:
                if index.flags() & Qt.ItemFlag.ItemIsEditable:
                    return self._editable_item_brush
            elif (
                role == Qt.ItemDataRole.UserRole
                and Column.from_index(index.column()) is Column.RELATIVE_POWER
            ):
                probe = try_get_probe(item)
                if probe is None:
                    return None
                return calc_coherent_percent(probe)

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        value = super().flags(index)

        if index.isValid():
            parent = index.parent()

            # Only item rows edit; a mode row has no editable column of its own.
            if not parent.isValid() and Column.from_index(index.column()).editable:
                value |= Qt.ItemFlag.ItemIsEditable

        return value

    def setData(self, index: QModelIndex, value: Any, role: int = Qt.ItemDataRole.EditRole) -> bool:  # noqa: N802
        if index.isValid() and role == Qt.ItemDataRole.EditRole:
            parent = index.parent()

            if not parent.isValid():
                column = Column.from_index(index.column())

                if column is Column.NAME:
                    self._repository.set_name(index.row(), str(value))
                    return True
                elif column is Column.BUILDER:
                    self._api.build_probe(index.row(), str(value))
                    return True

        return False

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        if parent.column() > 0:
            return 0

        node = parent.internalPointer() if parent.isValid() else self._tree_root
        return len(node.children)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(Column)

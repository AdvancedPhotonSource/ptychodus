from __future__ import annotations
from enum import IntEnum
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


class Column(IntEnum):
    """Column order of ProbeTreeModel.

    An enum rather than module-level ints so that ``match`` arms are value patterns:
    a bare name would be a capture pattern that swallows every column. Public because
    the controller installs item delegates on some of these columns.
    """

    NAME = 0
    RELATIVE_POWER = 1
    BUILDER = 2
    DATA_TYPE = 3
    WIDTH_PX = 4
    HEIGHT_PX = 5
    PIXEL_WIDTH_NM = 6
    PIXEL_HEIGHT_NM = 7
    SIZE = 8


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
        self._header = [
            'Name',
            'Relative Power',
            'Builder',
            'Data Type',
            'Width [px]',
            'Height [px]',
            'Pixel Width\n[nm]',
            'Pixel Height\n[nm]',
            'Size',
        ]

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
        top_left = self.index(index, 0)
        bottom_right = self.index(index, len(self._header))
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

        child_top_left = self.index(0, 0, top_left)
        child_bottom_right = self.index(num_modes_new, len(self._header), top_left)
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
            return self._header[section]

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
                match index.column():
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
            elif role == Qt.ItemDataRole.UserRole and index.column() == Column.RELATIVE_POWER:
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
                match index.column():
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
            elif role == Qt.ItemDataRole.UserRole and index.column() == Column.RELATIVE_POWER:
                probe = try_get_probe(item)
                if probe is None:
                    return None
                return calc_coherent_percent(probe)

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        value = super().flags(index)

        if index.isValid():
            parent = index.parent()

            if not parent.isValid() and index.column() in (Column.NAME, Column.BUILDER):
                value |= Qt.ItemFlag.ItemIsEditable

        return value

    def setData(self, index: QModelIndex, value: Any, role: int = Qt.ItemDataRole.EditRole) -> bool:  # noqa: N802
        if index.isValid() and role == Qt.ItemDataRole.EditRole:
            parent = index.parent()

            if not parent.isValid():
                if index.column() == Column.NAME:
                    self._repository.set_name(index.row(), str(value))
                    return True
                elif index.column() == Column.BUILDER:
                    self._api.build_probe(index.row(), str(value))
                    return True

        return False

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        if parent.column() > 0:
            return 0

        node = parent.internalPointer() if parent.isValid() else self._tree_root
        return len(node.children)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return len(self._header)

from typing import Callable

from PyQt5.QtCore import QMetaObject, QModelIndex, QSortFilterProxyModel, Qt, pyqtBoundSignal
from PyQt5.QtGui import QBrush, QPalette
from PyQt5.QtWidgets import QAbstractItemView, QAction, QWidget


def connect_current_changed_signal(
    view: QAbstractItemView, slot: Callable[..., None] | pyqtBoundSignal
) -> None:
    selection_model = view.selectionModel()

    if selection_model is None:
        raise ValueError('selection_model is None!')

    selection_model.currentChanged.connect(slot)


def connect_triggered_signal(
    action: QAction | None, slot: Callable[..., None] | pyqtBoundSignal
) -> QMetaObject.Connection:
    if action is None:
        raise ValueError('action is None!')

    return action.triggered.connect(slot)


def create_brush_for_editable_cell(widget: QWidget) -> QBrush:
    palette = widget.palette()
    alternate_base_color = palette.color(QPalette.AlternateBase)
    return QBrush(alternate_base_color)


class UserRoleSortFilterProxyModel(QSortFilterProxyModel):
    """Sorts on Qt.ItemDataRole.UserRole for columns that supply one; defers to the base otherwise.

    Columns whose display text is not sortable as a string -- an adaptive byte size such as
    "940 B" against "4.10 GB" -- return the underlying number under UserRole. Columns that
    supply no UserRole are unaffected and keep the base class ordering.
    """

    def lessThan(self, left: QModelIndex, right: QModelIndex) -> bool:  # noqa: N802
        lhs = left.data(Qt.ItemDataRole.UserRole)
        rhs = right.data(Qt.ItemDataRole.UserRole)

        if lhs is not None and rhs is not None:
            return bool(lhs < rhs)

        return super().lessThan(left, right)

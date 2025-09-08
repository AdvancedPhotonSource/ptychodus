from typing import Callable

from PyQt5.QtCore import QMetaObject, pyqtBoundSignal
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

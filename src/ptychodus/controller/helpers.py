from typing import Callable

from PyQt5.QtCore import QMetaObject, pyqtBoundSignal
from PyQt5.QtWidgets import QAction


def connect_triggered_signal(
    action: QAction | None, slot: Callable[..., None] | pyqtBoundSignal
) -> QMetaObject.Connection:
    if action is None:
        raise ValueError('QAction is None!')

    return action.triggered.connect(slot)

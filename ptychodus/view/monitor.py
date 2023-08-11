from __future__ import annotations
from typing import Optional

from PyQt5.QtWidgets import QVBoxLayout, QWidget, QStatusBar

from .image import ImageView
from .widgets import BottomTitledGroupBox


class MonitorProbeView(BottomTitledGroupBox):

    def __init__(self, statusbar: QStatusBar, parent: Optional[QWidget]) -> None:
        super().__init__('Probe', parent)
        self.imageView = ImageView.createInstance(statusbar)

    @classmethod
    def createInstance(cls, statusbar: QStatusBar, parent: Optional[QWidget] = None) -> MonitorProbeView:
        view = cls(statusbar, parent)

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 30)
        layout.addWidget(view.imageView)
        view.setLayout(layout)

        return view


class MonitorObjectView(BottomTitledGroupBox):

    def __init__(self, statusbar: QStatusBar, parent: Optional[QWidget]) -> None:
        super().__init__('Object', parent)
        self.imageView = ImageView.createInstance(statusbar)

    @classmethod
    def createInstance(cls, statusbar: QStatusBar, parent: Optional[QWidget] = None) -> MonitorObjectView:
        view = cls(statusbar, parent)

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 30)
        layout.addWidget(view.imageView)
        view.setLayout(layout)

        return view

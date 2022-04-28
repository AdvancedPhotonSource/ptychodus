from __future__ import annotations
from typing import Optional

from PyQt5.QtWidgets import QVBoxLayout, QWidget

from .image import ImageView
from .widgets import BottomTitledGroupBox


class MonitorProbeView(BottomTitledGroupBox):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__('Probe', parent)
        self.imageView = ImageView.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> MonitorProbeView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 30)
        layout.addWidget(view.imageView)
        view.setLayout(layout)
        view.imageView.imageRibbon.frameGroupBox.setVisible(False)  # TODO to controller

        return view


class MonitorObjectView(BottomTitledGroupBox):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__('Object', parent)
        self.imageView = ImageView.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> MonitorObjectView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 30)
        layout.addWidget(view.imageView)
        view.setLayout(layout)
        view.imageView.imageRibbon.frameGroupBox.setVisible(False)  # TODO to controller

        return view

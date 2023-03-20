from __future__ import annotations
from typing import Optional

from PyQt5.QtWidgets import (QFormLayout, QGroupBox, QVBoxLayout, QWidget)

from .widgets import LengthWidget, RepositoryWidget


class ObjectView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Parameters', parent)
        self.pixelSizeXWidget = LengthWidget.createInstance()
        self.pixelSizeYWidget = LengthWidget.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ObjectView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow('Pixel Size X:', view.pixelSizeXWidget)
        layout.addRow('Pixel Size Y:', view.pixelSizeYWidget)
        view.setLayout(layout)

        return view


class ObjectParametersView(QWidget):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.objectView = ObjectView.createInstance()
        self.repositoryWidget = RepositoryWidget.createInstance('Object Estimates')

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ObjectParametersView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.objectView)
        layout.addWidget(view.repositoryWidget)
        view.setLayout(layout)

        return view

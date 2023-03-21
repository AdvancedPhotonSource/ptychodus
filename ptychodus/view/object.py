from __future__ import annotations
from typing import Optional

from PyQt5.QtWidgets import (QFormLayout, QGroupBox, QVBoxLayout, QWidget)

from .widgets import LengthWidget, RepositoryView


class ObjectParametersView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Parameters', parent)
        self.pixelSizeXWidget = LengthWidget.createInstance()
        self.pixelSizeYWidget = LengthWidget.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ObjectParametersView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow('Pixel Size X:', view.pixelSizeXWidget)
        layout.addRow('Pixel Size Y:', view.pixelSizeYWidget)
        view.setLayout(layout)

        return view


class ObjectView(QWidget):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.parametersView = ObjectParametersView.createInstance()
        self.repositoryView = RepositoryView.createInstance('Repository')

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ObjectView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.parametersView)
        layout.addWidget(view.repositoryView)
        view.setLayout(layout)

        return view

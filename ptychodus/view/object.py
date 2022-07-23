from __future__ import annotations
from typing import Optional

from PyQt5.QtWidgets import (QComboBox, QFormLayout, QGroupBox, QPushButton, QSpinBox, QVBoxLayout,
                             QWidget)

from .widgets import LengthWidget


class ObjectInitializerView(QGroupBox):

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__('Initializer', parent)
        self.initializerComboBox = QComboBox()
        self.initializeButton = QPushButton('Initialize')

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ObjectInitializerView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.initializerComboBox)
        layout.addWidget(view.initializeButton)
        view.setLayout(layout)

        return view


class ObjectView(QGroupBox):

    def __init__(self, parent: Optional[QWidget] = None) -> None:
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

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.initializerView = ObjectInitializerView.createInstance()
        self.objectView = ObjectView.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ObjectParametersView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.initializerView)
        layout.addWidget(view.objectView)
        layout.addStretch()
        view.setLayout(layout)

        return view

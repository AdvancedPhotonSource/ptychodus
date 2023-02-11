from __future__ import annotations
from typing import Optional

from PyQt5.QtWidgets import (QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QMenu, QPushButton,
                             QSpinBox, QVBoxLayout, QWidget)

from .widgets import LengthWidget


class ObjectView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Parameters', parent)
        self.numberOfPixelsXSpinBox = QSpinBox()
        self.numberOfPixelsYSpinBox = QSpinBox()
        self.pixelSizeXWidget = LengthWidget.createInstance()
        self.pixelSizeYWidget = LengthWidget.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ObjectView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow('Number of Pixels X:', view.numberOfPixelsXSpinBox)
        layout.addRow('Number of Pixels Y:', view.numberOfPixelsYSpinBox)
        layout.addRow('Pixel Size X:', view.pixelSizeXWidget)
        layout.addRow('Pixel Size Y:', view.pixelSizeYWidget)
        view.setLayout(layout)

        return view


class ObjectButtonBox(QWidget):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.initializeMenu = QMenu()
        self.initializeButton = QPushButton('Initialize')
        self.saveButton = QPushButton('Save')

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ObjectButtonBox:
        view = cls(parent)

        view.initializeButton.setMenu(view.initializeMenu)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(view.initializeButton)
        layout.addWidget(view.saveButton)
        view.setLayout(layout)

        return view


class ObjectInitializerView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Initializer', parent)
        self.buttonBox = ObjectButtonBox.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ObjectInitializerView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.buttonBox)
        view.setLayout(layout)

        return view


class ObjectParametersView(QWidget):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.objectView = ObjectView.createInstance()
        self.initializerView = ObjectInitializerView.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ObjectParametersView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.objectView)
        layout.addWidget(view.initializerView)
        layout.addStretch()
        view.setLayout(layout)

        return view

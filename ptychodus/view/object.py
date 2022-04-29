from __future__ import annotations
from typing import Optional

from PyQt5.QtWidgets import QComboBox, QFormLayout, QGroupBox, QPushButton, QSpinBox, QVBoxLayout, QWidget

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


class ObjectParametersView(QWidget):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.initializerView = ObjectInitializerView.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> ObjectParametersView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.initializerView)
        layout.addStretch()
        view.setLayout(layout)

        return view

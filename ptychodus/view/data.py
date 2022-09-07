from __future__ import annotations
from typing import Optional

from PyQt5.QtWidgets import (QGroupBox, QHeaderView, QHBoxLayout, QTreeView, QPushButton,
                             QVBoxLayout, QWidget)


class DataButtonBox(QWidget):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.openButton = QPushButton('Open')
        self.saveButton = QPushButton('Save')

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> DataButtonBox:
        view = cls(parent)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(view.openButton)
        layout.addWidget(view.saveButton)
        view.setLayout(layout)

        return view


class DataView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Diffraction Data', parent)
        self.treeView = QTreeView()
        self.buttonBox = DataButtonBox.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> DataView:
        view = cls(parent)

        view.treeView.header().setSectionResizeMode(QHeaderView.ResizeToContents)

        layout = QVBoxLayout()
        layout.addWidget(view.treeView)
        layout.addWidget(view.buttonBox)
        view.setLayout(layout)

        return view


class DataParametersView(QWidget):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.dataView = DataView.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> DataParametersView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.dataView)
        view.setLayout(layout)

        return view

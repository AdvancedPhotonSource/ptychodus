from __future__ import annotations
from typing import Optional

from PyQt5.QtWidgets import (QGridLayout, QGroupBox, QLabel, QLineEdit, QPushButton, QSpinBox,
                             QVBoxLayout, QWidget)


class PtychoNNBasicParametersView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('PtychoNN Parameters', parent)
        self.modelStateLabel = QLabel('Model State:')
        self.modelStateLineEdit = QLineEdit()
        self.modelStateBrowseButton = QPushButton('Browse')
        self.batchSizeLabel = QLabel('Batch Size:')
        self.batchSizeSpinBox = QSpinBox()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> PtychoNNBasicParametersView:
        view = cls(parent)

        layout = QGridLayout()
        layout.addWidget(view.modelStateLabel, 0, 0)
        layout.addWidget(view.modelStateLineEdit, 0, 1)
        layout.addWidget(view.modelStateBrowseButton, 0, 2)
        layout.addWidget(view.batchSizeLabel, 1, 0)
        layout.addWidget(view.batchSizeSpinBox, 1, 1, 1, 2)
        layout.setColumnStretch(1, 1)
        view.setLayout(layout)

        return view


class PtychoNNParametersView(QWidget):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.basicParametersView = PtychoNNBasicParametersView.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> PtychoNNParametersView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.basicParametersView)
        layout.addStretch()
        view.setLayout(layout)

        return view

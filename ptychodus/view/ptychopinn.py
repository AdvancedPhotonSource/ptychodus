from __future__ import annotations
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QCheckBox, QFormLayout, QGridLayout, QGroupBox, QLabel, QLineEdit,
                             QPushButton, QSpinBox, QVBoxLayout, QWidget)

from .widgets import DecimalLineEdit, DecimalSlider


class PtychoPINNModelParametersView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Model Parameters', parent)
        # Assuming similar widgets are needed, adjust as per PtychoPINN specifics
        self.modelStateLabel = QLabel('Model State:')
        self.modelStateLineEdit = QLineEdit()
        self.modelStateBrowseButton = QPushButton('Browse')
        self.gridSizeLabel = QLabel('Grid Size:')
        self.gridSizeSpinBox = QSpinBox()
        self.batchSizeLabel = QLabel('Batch Size:')
        self.batchSizeSpinBox = QSpinBox()
        self.probeScaleLabel = QLabel('Probe Scale:')
        self.probeScaleLineEdit = DecimalLineEdit.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> PtychoPINNModelParametersView:
        view = cls(parent)

        layout = QGridLayout()
        layout.addWidget(view.modelStateLabel, 0, 0)
        layout.addWidget(view.modelStateLineEdit, 0, 1)
        layout.addWidget(view.modelStateBrowseButton, 0, 2)
        layout.addWidget(view.gridSizeLabel, 1, 0)
        layout.addWidget(view.gridSizeSpinBox, 1, 1, 1, 2)
        layout.addWidget(view.batchSizeLabel, 2, 0)
        layout.addWidget(view.batchSizeSpinBox, 2, 1, 1, 2)
        layout.addWidget(view.probeScaleLabel, 3, 0)
        layout.addWidget(view.probeScaleLineEdit, 3, 1, 1, 2)
        layout.setColumnStretch(1, 1)
        view.setLayout(layout)

        return view


class PtychoPINNOutputParametersView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Save Training Artifacts', parent)
        self.pathLabel = QLabel('Path:')
        self.pathLineEdit = QLineEdit()
        self.pathBrowseButton = QPushButton('Browse')

        self.suffixLabel = QLabel('Suffix:')
        self.suffixLineEdit = QLineEdit()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> \
            PtychoPINNOutputParametersView:
        view = cls(parent)

        layout = QGridLayout()
        layout.addWidget(view.pathLabel, 0, 0)
        layout.addWidget(view.pathLineEdit, 0, 1)
        layout.addWidget(view.pathBrowseButton, 0, 2)
        layout.addWidget(view.suffixLabel, 1, 0)
        layout.addWidget(view.suffixLineEdit, 1, 1, 1, 2)
        layout.setColumnStretch(1, 1)
        view.setLayout(layout)

        return view


class PtychoPINNTrainingParametersView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Training Parameters', parent)
        # Assuming similar widgets are needed, adjust as per PtychoPINN specifics
        self.trainingEpochsLabel = QLabel('Training Epochs:')
        self.trainingEpochsSpinBox = QSpinBox()
        self.outputParametersView = PtychoPINNOutputParametersView.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> PtychoPINNTrainingParametersView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow(view.trainingEpochsLabel, view.trainingEpochsSpinBox)
        layout.addRow(view.outputParametersView)
        view.setLayout(layout)

        return view


class PtychoPINNParametersView(QWidget):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.modelParametersView = PtychoPINNModelParametersView.createInstance()
        self.trainingParametersView = PtychoPINNTrainingParametersView.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> PtychoPINNParametersView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.modelParametersView)
        layout.addWidget(view.trainingParametersView)
        layout.addStretch()
        view.setLayout(layout)

        return view

from __future__ import annotations
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QCheckBox, QFormLayout, QGridLayout, QGroupBox, QLabel, QLineEdit,
                             QPushButton, QSpinBox, QVBoxLayout, QWidget)

from .widgets import DecimalLineEdit, DecimalSlider


class PtychoNNModelParametersView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Model Parameters', parent)
        self.modelStateLabel = QLabel('Model State:')
        self.modelStateLineEdit = QLineEdit()
        self.modelStateBrowseButton = QPushButton('Browse')
        self.numberOfConvolutionChannelsLabel = QLabel('Convolution Channels:')
        self.numberOfConvolutionChannelsSpinBox = QSpinBox()
        self.batchSizeLabel = QLabel('Batch Size:')
        self.batchSizeSpinBox = QSpinBox()
        self.useBatchNormalizationCheckBox = QCheckBox('Use Batch Normalization')

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> PtychoNNModelParametersView:
        view = cls(parent)

        layout = QGridLayout()
        layout.addWidget(view.modelStateLabel, 0, 0)
        layout.addWidget(view.modelStateLineEdit, 0, 1)
        layout.addWidget(view.modelStateBrowseButton, 0, 2)
        layout.addWidget(view.numberOfConvolutionChannelsLabel, 1, 0)
        layout.addWidget(view.numberOfConvolutionChannelsSpinBox, 1, 1, 1, 2)
        layout.addWidget(view.batchSizeLabel, 2, 0)
        layout.addWidget(view.batchSizeSpinBox, 2, 1, 1, 2)
        layout.addWidget(view.useBatchNormalizationCheckBox, 3, 0, 1, 3)
        layout.setColumnStretch(1, 1)
        view.setLayout(layout)

        return view


class PtychoNNOutputParametersView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Save Training Artifacts', parent)
        self.pathLabel = QLabel('Path:')
        self.pathLineEdit = QLineEdit()
        self.pathBrowseButton = QPushButton('Browse')

        self.suffixLabel = QLabel('Suffix:')
        self.suffixLineEdit = QLineEdit()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> \
            PtychoNNOutputParametersView:
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


class PtychoNNTrainingParametersView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Training Parameters', parent)
        self.validationSetFractionalSizeSlider = DecimalSlider.createInstance(Qt.Horizontal)
        self.optimizationEpochsPerHalfCycleSpinBox = QSpinBox()
        self.maximumLearningRateLineEdit = DecimalLineEdit.createInstance()
        self.minimumLearningRateLineEdit = DecimalLineEdit.createInstance()
        self.trainingEpochsSpinBox = QSpinBox()
        self.statusIntervalSpinBox = QSpinBox()
        self.outputParametersView = PtychoNNOutputParametersView.createInstance()
        self.trainButton = QPushButton('Train')

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> PtychoNNTrainingParametersView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow('Validation Set Fractional Size:', view.validationSetFractionalSizeSlider)
        layout.addRow('Optimization Epochs Per Half Cycle:',
                      view.optimizationEpochsPerHalfCycleSpinBox)
        layout.addRow('Maximum Learning Rate:', view.maximumLearningRateLineEdit)
        layout.addRow('Minimum Learning Rate:', view.minimumLearningRateLineEdit)
        layout.addRow('Training Epochs:', view.trainingEpochsSpinBox)
        layout.addRow('Status Interval:', view.statusIntervalSpinBox)
        layout.addRow(view.outputParametersView)
        layout.addRow(view.trainButton)
        view.setLayout(layout)

        return view


class PtychoNNTrainingDataView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Training Data', parent)
        self.exportButton = QPushButton('Export')

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> PtychoNNTrainingDataView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow(view.exportButton)
        view.setLayout(layout)

        return view


class PtychoNNParametersView(QWidget):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.modelParametersView = PtychoNNModelParametersView.createInstance()
        self.trainingParametersView = PtychoNNTrainingParametersView.createInstance()
        self.trainingDataView = PtychoNNTrainingDataView.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> PtychoNNParametersView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.modelParametersView)
        layout.addWidget(view.trainingParametersView)
        layout.addWidget(view.trainingDataView)
        layout.addStretch()
        view.setLayout(layout)

        return view

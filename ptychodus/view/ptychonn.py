from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QCheckBox, QFormLayout, QGridLayout, QGroupBox, QLabel, QLineEdit,
                             QPushButton, QSpinBox, QVBoxLayout, QWidget)

from .widgets import DecimalLineEdit, DecimalSlider


class PtychoNNModelParametersView(QGroupBox):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__('Model Parameters', parent)
        self.modelStateLabel = QLabel('Model State:')
        self.modelStateLineEdit = QLineEdit()
        self.modelStateBrowseButton = QPushButton('Browse')
        self.numberOfConvolutionKernelsLabel = QLabel('Convolution Kernels:')
        self.numberOfConvolutionKernelsSpinBox = QSpinBox()
        self.batchSizeLabel = QLabel('Batch Size:')
        self.batchSizeSpinBox = QSpinBox()
        self.useBatchNormalizationCheckBox = QCheckBox('Use Batch Normalization')

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> PtychoNNModelParametersView:
        view = cls(parent)

        layout = QGridLayout()
        layout.addWidget(view.modelStateLabel, 0, 0)
        layout.addWidget(view.modelStateLineEdit, 0, 1)
        layout.addWidget(view.modelStateBrowseButton, 0, 2)
        layout.addWidget(view.numberOfConvolutionKernelsLabel, 1, 0)
        layout.addWidget(view.numberOfConvolutionKernelsSpinBox, 1, 1, 1, 2)
        layout.addWidget(view.batchSizeLabel, 2, 0)
        layout.addWidget(view.batchSizeSpinBox, 2, 1, 1, 2)
        layout.addWidget(view.useBatchNormalizationCheckBox, 3, 0, 1, 3)
        layout.setColumnStretch(1, 1)
        view.setLayout(layout)

        return view


class PtychoNNTrainingArtifactsView(QGroupBox):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__('Save Training Artifacts', parent)
        self.directoryLabel = QLabel('Directory:')
        self.directoryLineEdit = QLineEdit()
        self.directoryBrowseButton = QPushButton('Browse')

        self.suffixLabel = QLabel('Suffix:')
        self.suffixLineEdit = QLineEdit()

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> \
            PtychoNNTrainingArtifactsView:
        view = cls(parent)

        layout = QGridLayout()
        layout.addWidget(view.directoryLabel, 0, 0)
        layout.addWidget(view.directoryLineEdit, 0, 1)
        layout.addWidget(view.directoryBrowseButton, 0, 2)
        layout.addWidget(view.suffixLabel, 1, 0)
        layout.addWidget(view.suffixLineEdit, 1, 1, 1, 2)
        layout.setColumnStretch(1, 1)
        view.setLayout(layout)

        return view


class PtychoNNTrainingParametersView(QGroupBox):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__('Training Parameters', parent)
        self.validationSetFractionalSizeSlider = DecimalSlider.createInstance(
            Qt.Orientation.Horizontal, numberOfTicks=20)
        self.maximumLearningRateLineEdit = DecimalLineEdit.createInstance()
        self.minimumLearningRateLineEdit = DecimalLineEdit.createInstance()
        self.trainingEpochsSpinBox = QSpinBox()
        self.statusIntervalSpinBox = QSpinBox()
        self.trainingArtifactsView = PtychoNNTrainingArtifactsView.createInstance()

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> PtychoNNTrainingParametersView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow('Validation Set Fractional Size:', view.validationSetFractionalSizeSlider)
        layout.addRow('Maximum Learning Rate:', view.maximumLearningRateLineEdit)
        layout.addRow('Minimum Learning Rate:', view.minimumLearningRateLineEdit)
        layout.addRow('Training Epochs:', view.trainingEpochsSpinBox)
        layout.addRow('Status Interval:', view.statusIntervalSpinBox)
        layout.addRow(view.trainingArtifactsView)
        view.setLayout(layout)

        return view


class PtychoNNParametersView(QWidget):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.modelParametersView = PtychoNNModelParametersView.createInstance()
        self.trainingParametersView = PtychoNNTrainingParametersView.createInstance()

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> PtychoNNParametersView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.modelParametersView)
        layout.addWidget(view.trainingParametersView)
        layout.addStretch()
        view.setLayout(layout)

        return view

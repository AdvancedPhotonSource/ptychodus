from __future__ import annotations
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QCheckBox, QFormLayout, QGridLayout, QGroupBox, QLabel, QLineEdit,
                             QPushButton, QSpinBox, QVBoxLayout, QWidget, QComboBox)

from .widgets import DecimalLineEdit, DecimalSlider


class PtychoNNModelParametersView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
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
    def createInstance(cls, parent: Optional[QWidget] = None) -> PtychoNNModelParametersView:
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
        self.validationSetFractionalSizeSlider = DecimalSlider.createInstance(
            Qt.Orientation.Horizontal)
        self.optimizationEpochsPerHalfCycleSpinBox = QSpinBox()
        self.maximumLearningRateLineEdit = DecimalLineEdit.createInstance()
        self.minimumLearningRateLineEdit = DecimalLineEdit.createInstance()
        self.trainingEpochsSpinBox = QSpinBox()
        self.statusIntervalSpinBox = QSpinBox()
        self.outputParametersView = PtychoNNOutputParametersView.createInstance()

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
        view.setLayout(layout)

        return view
    
class PtychoNNPositionPredictionParametersView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Position Prediction Parameters', parent)
        self.reconstructorImagePathLabel = QLabel('Reconstructor Image Path:')
        self.reconstructorImagePathLineEdit = QLineEdit()
        self.reconstructorImagePathBrowseButton = QPushButton('Browse')
        self.probePositionListPathLabel = QLabel('Probe Position List Path:')
        self.probePositionListPathLineEdit = QLineEdit()
        self.probePositionListPathBrowseButton = QPushButton('Browse')
        self.probePositionDataUnitLabel = QLabel('Probe Position Data Unit:')
        self.probePositionDataUnitLineEdit = QLineEdit()
        self.pixelSizeNMLabel = QLabel('Pixel Size NM:')
        self.pixelSizeNMLineEdit = QLineEdit()
        self.baselinePositionListLabel = QLabel('Baseline Position List:')
        self.baselinePositionListLineEdit = QLineEdit()
        self.baselinePositionListBrowseButton = QPushButton('Browse')
        self.centralCropLabel = QLabel('Central Crop:')
        self.centralCropLineEdit = QLineEdit()
        self.methodLabel = QLabel('Method:')
        self.methodLineEdit = QLineEdit()
        self.numberNeighborsCollectiveLabel = QLabel('Number of Neighbors Collective:')
        self.numberNeighborsCollectiveSpinbox = QSpinBox()
        self.offsetEstimatorOrderLabel = QLabel('Offset Estimator Order:')
        self.offsetEstimatorOrderLineEdit = QLineEdit()
        self.offsetEstimatorBetaLabel = QLabel('Offset Estimator Beta:')
        self.offsetEstimatorBetaLineEdit = QLineEdit()
        self.smoothConstraintWeightLabel = QLabel('Smooth Constraint Weight:')
        self.smoothConstraintWeightLineEdit = QLineEdit()
        self.rectangularGridLabel = QLabel('Rectangular Grid:')
        self.rectangularGridLineEdit = QLineEdit()
        self.randomSeedLabel = QLabel('Random Seed:')
        self.randomSeedLineEdit = QLineEdit()
        self.debugLabel = QLabel('Debug:')
        self.debugLineEdit = QLineEdit()
        self.registrationParametersView = PtychoNNRegistrationParametersView.createInstance()
        
        self.runButton = QPushButton('Run')
        self.reconstructButton = QPushButton('Reconstruct')

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> PtychoNNPositionPredictionParametersView:
        view = cls(parent)

        layout = QGridLayout()
        layout.addWidget(view.reconstructorImagePathLabel, 0, 0)
        layout.addWidget(view.reconstructorImagePathLineEdit, 0, 1)
        layout.addWidget(view.reconstructorImagePathBrowseButton, 0, 2)
        layout.addWidget(view.probePositionListPathLabel, 1, 0)
        layout.addWidget(view.probePositionListPathLineEdit, 1, 1)
        layout.addWidget(view.probePositionListPathBrowseButton, 1, 2)
        layout.addWidget(view.probePositionDataUnitLabel, 2, 0)
        layout.addWidget(view.probePositionDataUnitLineEdit, 2, 1, 1, 2)
        layout.addWidget(view.pixelSizeNMLabel, 3, 0)
        layout.addWidget(view.pixelSizeNMLineEdit, 3, 1, 1, 2)
        layout.addWidget(view.baselinePositionListLabel, 4, 0)
        layout.addWidget(view.baselinePositionListLineEdit, 4, 1)
        layout.addWidget(view.baselinePositionListBrowseButton, 4, 2)
        layout.addWidget(view.centralCropLabel, 5, 0)
        layout.addWidget(view.centralCropLineEdit, 5, 1, 1, 2)
        layout.addWidget(view.methodLabel, 6, 0)
        layout.addWidget(view.methodLineEdit, 6, 1, 1, 2)
        layout.addWidget(view.numberNeighborsCollectiveLabel, 7, 0)
        layout.addWidget(view.numberNeighborsCollectiveSpinbox, 7, 1, 1, 2)
        layout.addWidget(view.offsetEstimatorOrderLabel, 8, 0)
        layout.addWidget(view.offsetEstimatorOrderLineEdit, 8, 1, 1, 2)
        layout.addWidget(view.offsetEstimatorBetaLabel, 9, 0)
        layout.addWidget(view.offsetEstimatorBetaLineEdit, 9, 1, 1, 2)
        layout.addWidget(view.smoothConstraintWeightLabel, 10, 0)
        layout.addWidget(view.smoothConstraintWeightLineEdit, 10, 1, 1, 2)
        layout.addWidget(view.rectangularGridLabel, 11, 0)
        layout.addWidget(view.rectangularGridLineEdit, 11, 1, 1, 2)
        layout.addWidget(view.randomSeedLabel, 12, 0)
        layout.addWidget(view.randomSeedLineEdit, 12, 1, 1, 2)
        layout.addWidget(view.debugLabel, 13, 0)
        layout.addWidget(view.debugLineEdit, 13, 1, 1, 2)
        layout.addWidget(view.registrationParametersView, 14, 0, 1, 3)
        layout.addWidget(view.runButton, 15, 0)
        layout.addWidget(view.reconstructButton, 15, 1, 1, 2)

        layout.setColumnStretch(1, 1)
        view.setLayout(layout)
        
        return view

class PtychoNNRegistrationParametersView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Registration Parameters', parent)
        self.registrationParamsLabel = QLabel('Registration Parameters:')
        self.registrationMethodLabel = QLabel('Registration Method:')
        self.registrationMethodDropDown = QComboBox()
        self.hybridRegistrationAlgsLabel = QLabel('Hybrid Registration Algs:')
        self.hybridRegistrationLineEdit = QLineEdit()
        self.nonhybridRegistrationTolsLabel = QLabel('Nonhybrid Registration Tols:')
        self.nonhybridRegistrationTolsLineEdit = QLineEdit()
        self.maxShiftLabel = QLabel('Max Shift:')
        self.maxShiftLineEdit = QLineEdit()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> \
            PtychoNNRegistrationParametersView:
        view = cls(parent)

        layout = QGridLayout()
        layout.addWidget(view.registrationParamsLabel, 14, 0)
        layout.addWidget(view.registrationMethodLabel, 15, 0)
        layout.addWidget(view.registrationMethodDropDown, 15, 1, 1, 2)
        layout.addWidget(view.hybridRegistrationAlgsLabel, 16, 0)
        layout.addWidget(view.hybridRegistrationLineEdit, 16, 1, 1, 2)
        layout.addWidget(view.nonhybridRegistrationTolsLabel, 17, 0)
        layout.addWidget(view.nonhybridRegistrationTolsLineEdit, 17, 1, 1, 2)
        layout.addWidget(view.maxShiftLabel, 18, 0)
        layout.addWidget(view.maxShiftLineEdit, 18, 1, 1, 2)
        layout.setColumnStretch(1, 1)
        view.setLayout(layout)

        return view

class PtychoNNParametersView(QWidget):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.modelParametersView = PtychoNNModelParametersView.createInstance()
        self.trainingParametersView = PtychoNNTrainingParametersView.createInstance()
        self.positionPredictionParametersView = PtychoNNPositionPredictionParametersView.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> PtychoNNParametersView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.modelParametersView)
        layout.addWidget(view.trainingParametersView)
        layout.addWidget(view.positionPredictionParametersView)
        layout.addStretch()
        view.setLayout(layout)

        return view

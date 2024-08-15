from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QCheckBox, QFormLayout, QGroupBox, QSpinBox, QVBoxLayout, QWidget,
                             QLineEdit, QPushButton, QGridLayout, QLabel, QComboBox)

from .widgets import DecimalLineEdit, DecimalSlider


class PtychoNNModelParametersView(QGroupBox):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__('Model Parameters', parent)
        self.numberOfConvolutionKernelsSpinBox = QSpinBox()
        self.batchSizeSpinBox = QSpinBox()
        self.useBatchNormalizationCheckBox = QCheckBox('Use Batch Normalization')

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> PtychoNNModelParametersView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow('Convolution Kernels:', view.numberOfConvolutionKernelsSpinBox)
        layout.addRow('Batch Size:', view.batchSizeSpinBox)
        layout.addRow(view.useBatchNormalizationCheckBox)
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

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> PtychoNNTrainingParametersView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow('Validation Set Fractional Size:', view.validationSetFractionalSizeSlider)
        layout.addRow('Maximum Learning Rate:', view.maximumLearningRateLineEdit)
        layout.addRow('Minimum Learning Rate:', view.minimumLearningRateLineEdit)
        layout.addRow('Training Epochs:', view.trainingEpochsSpinBox)
        layout.addRow('Status Interval:', view.statusIntervalSpinBox)
        view.setLayout(layout)

        return view
    
class PtychoNNPositionPredictionParametersView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Position Prediction Parameters', parent)
        self.reconstructedImagePathLabel = QLabel('Reconstructed Image Path:')
        self.reconstructedImagePathLineEdit = QLineEdit()
        self.reconstructedImagePathBrowseButton = QPushButton('Browse')
        self.probePositionListPathLabel = QLabel('Probe Position List Path:')
        self.probePositionListPathLineEdit = QLineEdit()
        self.probePositionListPathBrowseButton = QPushButton('Browse')
        self.probePositionDataUnitLabel = QLabel('Probe Position Data Unit:')
        
        self.probePositionDataUnitDropDown = QComboBox()
        self.probePositionDataUnitDropDown.addItem('nm')
        self.probePositionDataUnitDropDown.addItem('m')
        self.probePositionDataUnitDropDown.addItem('pixel')
        
        self.pixelSizeNMLabel = QLabel('Pixel Size NM:')
        self.pixelSizeNMLineEdit = DecimalLineEdit.createInstance()
        self.baselinePositionListLabel = QLabel('Baseline Position List:')
        self.baselinePositionListPathLineEdit = QLineEdit()
        self.baselinePositionListBrowseButton = QPushButton('Browse')
        self.centralCropLabel = QLabel('Central Crop:')
        self.centralCropLineEdit = QLineEdit()
        self.methodLabel = QLabel('Method:')
        
        self.methodDropDown = QComboBox()
        self.methodDropDown.addItem('collective')
        self.methodDropDown.addItem('serial')
        
        self.numberNeighborsCollectiveLabel = QLabel('Number of Neighbors Collective:')
        self.numberNeighborsCollectiveSpinbox = QSpinBox()
        self.offsetEstimatorOrderLabel = QLabel('Offset Estimator Order:')
        self.offsetEstimatorOrderLineEdit = QLineEdit()
        self.offsetEstimatorBetaLabel = QLabel('Offset Estimator Beta:')
        self.offsetEstimatorBetaLineEdit = DecimalLineEdit.createInstance()
        self.smoothConstraintWeightLabel = QLabel('Smooth Constraint Weight:')
        self.smoothConstraintWeightLineEdit = DecimalLineEdit.createInstance()
        self.rectangularGridLabel = QLabel('Rectangular Grid:')
        self.rectangularCheckBox = QCheckBox()
        self.randomSeedLabel = QLabel('Random Seed:')
        self.randomSeedLineEdit = QLineEdit()
        self.debugLabel = QLabel('Debug:')
        self.debugCheckBox = QCheckBox()
        self.registrationParametersView = PtychoNNRegistrationParametersView.createInstance()
        
    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> PtychoNNPositionPredictionParametersView:
        view = cls(parent)

        layout = QGridLayout()
        layout.addWidget(view.reconstructedImagePathLabel, 0, 0)
        layout.addWidget(view.reconstructedImagePathLineEdit, 0, 1)
        layout.addWidget(view.reconstructedImagePathBrowseButton, 0, 2)
        layout.addWidget(view.probePositionListPathLabel, 1, 0)
        layout.addWidget(view.probePositionListPathLineEdit, 1, 1)
        layout.addWidget(view.probePositionListPathBrowseButton, 1, 2)
        layout.addWidget(view.probePositionDataUnitLabel, 2, 0)
        layout.addWidget(view.probePositionDataUnitDropDown, 2, 1, 1, 2)
        layout.addWidget(view.pixelSizeNMLabel, 3, 0)
        layout.addWidget(view.pixelSizeNMLineEdit, 3, 1, 1, 2)
        layout.addWidget(view.baselinePositionListLabel, 4, 0)
        layout.addWidget(view.baselinePositionListPathLineEdit, 4, 1)
        layout.addWidget(view.baselinePositionListBrowseButton, 4, 2)
        layout.addWidget(view.centralCropLabel, 5, 0)
        layout.addWidget(view.centralCropLineEdit, 5, 1, 1, 2)
        layout.addWidget(view.methodLabel, 6, 0)
        layout.addWidget(view.methodDropDown, 6, 1, 1, 2)
        layout.addWidget(view.numberNeighborsCollectiveLabel, 7, 0)
        layout.addWidget(view.numberNeighborsCollectiveSpinbox, 7, 1, 1, 2)
        layout.addWidget(view.offsetEstimatorOrderLabel, 8, 0)
        layout.addWidget(view.offsetEstimatorOrderLineEdit, 8, 1, 1, 2)
        layout.addWidget(view.offsetEstimatorBetaLabel, 9, 0)
        layout.addWidget(view.offsetEstimatorBetaLineEdit, 9, 1, 1, 2)
        layout.addWidget(view.smoothConstraintWeightLabel, 10, 0)
        layout.addWidget(view.smoothConstraintWeightLineEdit, 10, 1, 1, 2)
        layout.addWidget(view.rectangularGridLabel, 11, 0)
        layout.addWidget(view.rectangularCheckBox, 11, 1, 1, 2)
        layout.addWidget(view.randomSeedLabel, 12, 0)
        layout.addWidget(view.randomSeedLineEdit, 12, 1, 1, 2)
        layout.addWidget(view.debugLabel, 13, 0)
        layout.addWidget(view.debugCheckBox, 13, 1, 1, 2)
        layout.addWidget(view.registrationParametersView, 14, 0, 1, 3)

        layout.setColumnStretch(1, 1)
        view.setLayout(layout)
        
        return view

class PtychoNNRegistrationParametersView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Registration Parameters', parent)
        self.registrationParamsLabel = QLabel('Registration Parameters:')
        self.registrationMethodLabel = QLabel('Registration Method:')
        
        self.registrationMethodDropDown = QComboBox()
        self.registrationMethodDropDown.addItem('hybrid')
        self.registrationMethodDropDown.addItem('error_map')
        self.registrationMethodDropDown.addItem('sift')
        
        self.hybridRegistrationTolsLabel = QLabel('Hybrid Registration Tols:')
        self.hybridRegistrationTolsLineEdit = QLineEdit()
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
        layout.addWidget(view.hybridRegistrationTolsLabel, 16, 0)
        layout.addWidget(view.hybridRegistrationTolsLineEdit, 16, 1, 1, 2)
        layout.addWidget(view.nonhybridRegistrationTolsLabel, 17, 0)
        layout.addWidget(view.nonhybridRegistrationTolsLineEdit, 17, 1, 1, 2)
        layout.addWidget(view.maxShiftLabel, 18, 0)
        layout.addWidget(view.maxShiftLineEdit, 18, 1, 1, 2)
        layout.setColumnStretch(1, 1)
        view.setLayout(layout)

        return view

class PtychoNNParametersView(QWidget):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.modelParametersView = PtychoNNModelParametersView.createInstance()
        self.trainingParametersView = PtychoNNTrainingParametersView.createInstance()
        self.positionPredictionParametersView = PtychoNNPositionPredictionParametersView.createInstance()

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> PtychoNNParametersView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.modelParametersView)
        layout.addWidget(view.trainingParametersView)
        layout.addWidget(view.positionPredictionParametersView)
        layout.addStretch()
        view.setLayout(layout)

        return view

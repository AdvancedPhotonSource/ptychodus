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
        self.nFiltersScaleLabel = QLabel('N Filters Scale:')
        self.nFiltersScaleSpinBox = QSpinBox()
        self.nPhotonsLabel = QLabel('N Photons:')
        self.nPhotonsLineEdit = DecimalLineEdit.createInstance()
        self.probeTrainableCheckBox = QCheckBox('Probe Trainable')
        self.intensityScaleTrainableCheckBox = QCheckBox('Intensity Scale Trainable')
        self.objectBigCheckBox = QCheckBox('Object Big')
        self.probeBigCheckBox = QCheckBox('Probe Big')
        self.probeScaleLabel = QLabel('Probe Scale:')
        self.probeScaleLineEdit = DecimalLineEdit.createInstance()
        self.probeMaskCheckBox = QCheckBox('Probe Mask')
        self.ampActivationLabel = QLabel('Amp Activation:')
        self.ampActivationLineEdit = QLineEdit()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> PtychoPINNModelParametersView:
        view = cls(parent)

        layout = QGridLayout()
        layout.addWidget(view.modelStateLabel, 0, 0)
        layout.addWidget(view.modelStateLineEdit, 0, 1)
        layout.addWidget(view.modelStateBrowseButton, 0, 2)
        layout.addWidget(view.gridSizeLabel, 1, 0)
        layout.addWidget(view.gridSizeSpinBox, 1, 1, 1, 2)
        layout.addWidget(view.nFiltersScaleLabel, 2, 0)
        layout.addWidget(view.nFiltersScaleSpinBox, 2, 1, 1, 2)
        layout.addWidget(view.nPhotonsLabel, 3, 0)
        layout.addWidget(view.nPhotonsLineEdit, 3, 1, 1, 2)
        layout.addWidget(view.probeTrainableCheckBox, 4, 0, 1, 3)
        layout.addWidget(view.intensityScaleTrainableCheckBox, 5, 0, 1, 3)
        layout.addWidget(view.objectBigCheckBox, 6, 0, 1, 3)
        layout.addWidget(view.probeBigCheckBox, 7, 0, 1, 3)
        layout.addWidget(view.probeScaleLabel, 8, 0)
        layout.addWidget(view.probeScaleLineEdit, 8, 1, 1, 2)
        layout.addWidget(view.probeMaskCheckBox, 9, 0, 1, 3)
        layout.addWidget(view.ampActivationLabel, 9, 0)
        layout.addWidget(view.ampActivationLineEdit, 9, 1, 1, 2)
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
        self.maximumTrainingDatasetSizeLabel = QLabel('Maximum Training Dataset Size:')
        self.maximumTrainingDatasetSizeSpinBox = QSpinBox()
        self.validationSetFractionalSizeLabel = QLabel('Validation Set Fractional Size:')
        self.validationSetFractionalSizeSlider = DecimalSlider.createInstance(Qt.Horizontal)
        self.maximumLearningRateLabel = QLabel('Maximum Learning Rate:')
        self.maximumLearningRateLineEdit = DecimalLineEdit.createInstance()
        self.minimumLearningRateLabel = QLabel('Minimum Learning Rate:')
        self.minimumLearningRateLineEdit = DecimalLineEdit.createInstance()
        # Assuming similar widgets are needed, adjust as per PtychoPINN specifics
        self.trainingEpochsLabel = QLabel('Training Epochs:')
        self.trainingEpochsSpinBox = QSpinBox()
        self.maeWeightLabel = QLabel('MAE Weight:')
        self.maeWeightLineEdit = DecimalLineEdit.createInstance()
        self.nllWeightLabel = QLabel('NLL Weight:')
        self.nllWeightLineEdit = DecimalLineEdit.createInstance()
        self.realspaceMAEWeightLabel = QLabel('Realspace MAE Weight:')
        self.realspaceMAEWeightLineEdit = DecimalLineEdit.createInstance()
        self.realspaceWeightLabel = QLabel('Realspace Weight:')
        self.realspaceWeightLineEdit = DecimalLineEdit.createInstance()
        self.outputParametersView = PtychoPINNOutputParametersView.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> PtychoPINNTrainingParametersView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow(view.maximumTrainingDatasetSizeLabel, view.maximumTrainingDatasetSizeSpinBox)
        layout.addRow('Validation Set Fractional Size:', view.validationSetFractionalSizeSlider)
        layout.addRow('Maximum Learning Rate:', view.maximumLearningRateLineEdit)
        layout.addRow('Minimum Learning Rate:', view.minimumLearningRateLineEdit)
        layout.addRow(view.trainingEpochsLabel, view.trainingEpochsSpinBox)
        layout.addRow('MAE Weight:', view.maeWeightLineEdit)
        layout.addRow('NLL Weight:', view.nllWeightLineEdit)
        layout.addRow('Realspace MAE Weight:', view.realspaceMAEWeightLineEdit)
        layout.addRow('Realspace Weight:', view.realspaceWeightLineEdit)
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

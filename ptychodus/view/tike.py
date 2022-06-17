from __future__ import annotations
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QGroupBox, QLineEdit, QSpinBox,
                             QVBoxLayout, QWidget)

from .widgets import DecimalLineEdit, DecimalSlider


class TikeBasicParametersView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Basic Parameters', parent)
        self.useMpiCheckBox = QCheckBox('Use MPI')
        self.numGpusLineEdit = QLineEdit()
        self.noiseModelComboBox = QComboBox()
        self.numProbeModesSpinBox = QSpinBox()
        self.numBatchSpinBox = QSpinBox()
        self.numIterSpinBox = QSpinBox()
        self.cgIterSpinBox = QSpinBox()
        self.alphaSlider = DecimalSlider.createInstance(Qt.Horizontal)
        self.stepLengthSlider = DecimalSlider.createInstance(Qt.Horizontal)

    @classmethod
    def createInstance(cls,
                       showCgIter: bool,
                       showAlpha: bool,
                       showStepLength: bool,
                       parent: Optional[QWidget] = None) -> TikeBasicParametersView:
        view = cls(parent)

        view.useMpiCheckBox.setToolTip('Whether to use MPI or not.')
        view.numGpusLineEdit.setToolTip(
            'The number of GPUs to use. If the number of GPUs is less than the requested number, only workers for the available GPUs are allocated.'
        )
        view.noiseModelComboBox.setToolTip('The noise model to use for the cost function.')
        view.numProbeModesSpinBox.setToolTip(
            'Number of orthogonal probe modes to simulate partial incoherence of the beam')
        view.numBatchSpinBox.setToolTip(
            'The dataset is divided into this number of groups where each group is processed sequentially.'
        )
        view.numIterSpinBox.setToolTip('The number of epochs to process before returning.')
        view.cgIterSpinBox.setToolTip(
            'The number of conjugate directions to search for each update.')
        view.alphaSlider.setToolTip('RPIE becomes EPIE when this parameter is 1.')
        view.stepLengthSlider.setToolTip(
            'Scales the inital search directions before the line search.')

        layout = QFormLayout()
        layout.addRow(view.useMpiCheckBox)
        layout.addRow('Number of GPUs:', view.numGpusLineEdit)
        layout.addRow('Noise Model:', view.noiseModelComboBox)
        layout.addRow('Number of Probe Modes:', view.numProbeModesSpinBox)
        layout.addRow('Number of Batches:', view.numBatchSpinBox)
        layout.addRow('Number of Iterations:', view.numIterSpinBox)

        if showCgIter:
            layout.addRow('CG Search Directions:', view.cgIterSpinBox)

        if showAlpha:
            layout.addRow('Alpha:', view.alphaSlider)

        if showStepLength:
            layout.addRow('Step Length:', view.stepLengthSlider)

        view.setLayout(layout)

        return view


class TikeAdaptiveMomentView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Use Adaptive Moment', parent)
        self.mdecaySlider = DecimalSlider.createInstance(Qt.Horizontal)
        self.vdecaySlider = DecimalSlider.createInstance(Qt.Horizontal)

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> TikeAdaptiveMomentView:
        view = cls(parent)

        view.mdecaySlider.setToolTip('The proportion of the first moment '
                                     'that is previous first moments.')
        view.vdecaySlider.setToolTip('The proportion of the second moment '
                                     'that is previous second moments.')

        layout = QFormLayout()
        layout.addRow('M Decay:', view.mdecaySlider)
        layout.addRow('V Decay:', view.vdecaySlider)
        view.setLayout(layout)

        return view


class TikePositionCorrectionView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Position Correction', parent)
        self.positionRegularizationCheckBox = QCheckBox('Use Regularization')
        self.adaptiveMomentView = TikeAdaptiveMomentView.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> TikePositionCorrectionView:
        view = cls(parent)

        view.positionRegularizationCheckBox.setToolTip(
            'Whether the positions are constrained to fit a random error plus affine error model.')

        layout = QFormLayout()
        layout.addRow(view.positionRegularizationCheckBox)
        layout.addRow(view.adaptiveMomentView)
        view.setLayout(layout)

        return view


class TikeProbeSupportView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Finite Probe Support', parent)
        self.weightLineEdit = DecimalLineEdit.createInstance()
        self.radiusSlider = DecimalSlider.createInstance(Qt.Horizontal)
        self.degreeLineEdit = DecimalLineEdit.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> TikeProbeSupportView:
        view = cls(parent)

        view.weightLineEdit.setToolTip('Weight of the finite probe constraint.')
        view.radiusSlider.setToolTip('Radius of probe support as fraction of probe grid.')
        view.degreeLineEdit.setToolTip('Degree of the supergaussian defining the probe support.')

        layout = QFormLayout()
        layout.addRow('Weight:', view.weightLineEdit)
        layout.addRow('Radius:', view.radiusSlider)
        layout.addRow('Degree:', view.degreeLineEdit)
        view.setLayout(layout)

        return view


class TikeProbeCorrectionView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Probe Correction', parent)
        self.sparsityConstraintSlider = DecimalSlider.createInstance(Qt.Horizontal)
        self.orthogonalityConstraintCheckBox = QCheckBox('Orthogonality Constraint')
        self.centeredIntensityConstraintCheckBox = QCheckBox('Centered Intensity Constraint')
        self.probeSupportView = TikeProbeSupportView.createInstance()
        self.adaptiveMomentView = TikeAdaptiveMomentView.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> TikeProbeCorrectionView:
        view = cls(parent)

        view.sparsityConstraintSlider.setToolTip(
            'Forces a maximum proportion of non-zero elements.')
        view.orthogonalityConstraintCheckBox.setToolTip(
            'Forces probes to be orthogonal each iteration.')
        view.centeredIntensityConstraintCheckBox.setToolTip(
            'Forces the probe intensity to be centered.')

        layout = QFormLayout()
        layout.addRow('Sparsity Constraint:', view.sparsityConstraintSlider)
        layout.addRow(view.orthogonalityConstraintCheckBox)
        layout.addRow(view.centeredIntensityConstraintCheckBox)
        layout.addRow(view.probeSupportView)
        layout.addRow(view.adaptiveMomentView)
        view.setLayout(layout)

        return view


class TikeObjectCorrectionView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Object Correction', parent)
        self.positivityConstraintSlider = DecimalSlider.createInstance(Qt.Horizontal)
        self.smoothnessConstraintSlider = DecimalSlider.createInstance(Qt.Horizontal)
        self.adaptiveMomentView = TikeAdaptiveMomentView.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> TikeObjectCorrectionView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow('Positivity Constraint:', view.positivityConstraintSlider)
        layout.addRow('Smoothness Constraint:', view.smoothnessConstraintSlider)
        layout.addRow(view.adaptiveMomentView)
        view.setLayout(layout)

        return view


class TikeParametersView(QWidget):

    def __init__(self, showCgIter: bool, showAlpha: bool, showStepLength: bool,
                 parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.basicParametersView = TikeBasicParametersView.createInstance(
            showCgIter, showAlpha, showStepLength)
        self.positionCorrectionView = TikePositionCorrectionView.createInstance()
        self.probeCorrectionView = TikeProbeCorrectionView.createInstance()
        self.objectCorrectionView = TikeObjectCorrectionView.createInstance()

    @classmethod
    def createInstance(cls,
                       showCgIter: bool,
                       showAlpha: bool,
                       showStepLength: bool,
                       parent: Optional[QWidget] = None) -> TikeParametersView:
        view = cls(showCgIter, showAlpha, showStepLength, parent)

        layout = QVBoxLayout()
        layout.addWidget(view.basicParametersView)
        layout.addWidget(view.positionCorrectionView)
        layout.addWidget(view.probeCorrectionView)
        layout.addWidget(view.objectCorrectionView)
        layout.addStretch()
        view.setLayout(layout)

        return view

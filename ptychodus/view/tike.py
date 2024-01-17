from __future__ import annotations
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QGroupBox, QLineEdit, QSpinBox,
                             QVBoxLayout, QWidget)

from .widgets import DecimalLineEdit, DecimalSlider


class TikeBasicParametersView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Tike Parameters', parent)
        self.numGpusLineEdit = QLineEdit()
        self.noiseModelComboBox = QComboBox()
        self.numBatchSpinBox = QSpinBox()
        self.batchMethodComboBox = QComboBox()
        self.numIterSpinBox = QSpinBox()
        self.convergenceWindowSpinBox = QSpinBox()
        self.alphaSlider = DecimalSlider.createInstance(Qt.Orientation.Horizontal)
        self.stepLengthSlider = DecimalSlider.createInstance(Qt.Orientation.Horizontal)
        self.logLevelComboBox = QComboBox()

    @classmethod
    def createInstance(cls,
                       showAlpha: bool,
                       showStepLength: bool,
                       parent: Optional[QWidget] = None) -> TikeBasicParametersView:
        view = cls(parent)

        view.numGpusLineEdit.setToolTip(
            'The number of GPUs to use. If the number of GPUs is less than the requested number, '
            'only workers for the available GPUs are allocated.')
        view.noiseModelComboBox.setToolTip('The noise model to use for the cost function.')
        view.numBatchSpinBox.setToolTip('The dataset is divided into this number of groups '
                                        'where each group is processed sequentially.')
        view.batchMethodComboBox.setToolTip('The name of the batch selection method.')
        view.numIterSpinBox.setToolTip('The number of epochs to process before returning.')
        view.convergenceWindowSpinBox.setToolTip(
            'The number of epochs to consider for convergence monitoring. '
            'Set to any value less than 2 to disable.')
        view.alphaSlider.setToolTip('RPIE becomes EPIE when this parameter is 1.')
        view.stepLengthSlider.setToolTip(
            'Scales the inital search directions before the line search.')

        layout = QFormLayout()
        layout.addRow('Number of GPUs:', view.numGpusLineEdit)
        layout.addRow('Noise Model:', view.noiseModelComboBox)
        layout.addRow('Number of Batches:', view.numBatchSpinBox)
        layout.addRow('Batch Method:', view.batchMethodComboBox)
        layout.addRow('Number of Iterations:', view.numIterSpinBox)
        layout.addRow('Convergence Window:', view.convergenceWindowSpinBox)

        if showAlpha:
            layout.addRow('Alpha:', view.alphaSlider)

        if showStepLength:
            layout.addRow('Step Length:', view.stepLengthSlider)

        layout.addRow('Log Level:', view.logLevelComboBox)

        view.setLayout(layout)

        return view


class TikeAdaptiveMomentView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Adaptive Moment', parent)
        self.mdecaySlider = DecimalSlider.createInstance(Qt.Orientation.Horizontal)
        self.vdecaySlider = DecimalSlider.createInstance(Qt.Orientation.Horizontal)

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


class TikeMultigridView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Multigrid', parent)
        self.numLevelsSpinBox = QSpinBox()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> TikeMultigridView:
        view = cls(parent)

        view.numLevelsSpinBox.setToolTip(
            'The number of times to reduce the problem by a factor of two.')

        layout = QFormLayout()
        layout.addRow('Number of Levels:', view.numLevelsSpinBox)
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
        self.radiusSlider = DecimalSlider.createInstance(Qt.Orientation.Horizontal)
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
        self.forceSparsitySlider = DecimalSlider.createInstance(Qt.Orientation.Horizontal)
        self.forceOrthogonalityCheckBox = QCheckBox('Force Orthogonality')
        self.forceCenteredIntensityCheckBox = QCheckBox('Force Centered Intensity')
        self.probeSupportView = TikeProbeSupportView.createInstance()
        self.adaptiveMomentView = TikeAdaptiveMomentView.createInstance()
        self.additionalProbePenaltyLineEdit = DecimalLineEdit.createInstance()

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> TikeProbeCorrectionView:
        view = cls(parent)

        view.forceSparsitySlider.setToolTip('Forces this proportion of zero elements.')
        view.forceOrthogonalityCheckBox.setToolTip(
            'Forces probes to be orthogonal each iteration.')
        view.forceCenteredIntensityCheckBox.setToolTip(
            'Forces the probe intensity to be centered.')
        view.additionalProbePenaltyLineEdit.setToolTip(
            'Penalty applied to the last probe for existing.')

        layout = QFormLayout()
        layout.addRow('Force Sparsity:', view.forceSparsitySlider)
        layout.addRow(view.forceOrthogonalityCheckBox)
        layout.addRow(view.forceCenteredIntensityCheckBox)
        layout.addRow(view.probeSupportView)
        layout.addRow(view.adaptiveMomentView)
        layout.addRow('Additional Probe Penalty:', view.additionalProbePenaltyLineEdit)
        view.setLayout(layout)

        return view


class TikeObjectCorrectionView(QGroupBox):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__('Object Correction', parent)
        self.positivityConstraintSlider = DecimalSlider.createInstance(Qt.Orientation.Horizontal)
        self.smoothnessConstraintSlider = DecimalSlider.createInstance(Qt.Orientation.Horizontal)
        self.adaptiveMomentView = TikeAdaptiveMomentView.createInstance()
        self.useMagnitudeClippingCheckBox = QCheckBox('Magnitude Clipping')

    @classmethod
    def createInstance(cls, parent: Optional[QWidget] = None) -> TikeObjectCorrectionView:
        view = cls(parent)

        view.useMagnitudeClippingCheckBox.setToolTip('Forces the object magnitude to be <= 1.')

        layout = QFormLayout()
        layout.addRow('Positivity Constraint:', view.positivityConstraintSlider)
        layout.addRow('Smoothness Constraint:', view.smoothnessConstraintSlider)
        layout.addRow(view.adaptiveMomentView)
        layout.addRow(view.useMagnitudeClippingCheckBox)
        view.setLayout(layout)

        return view


class TikeParametersView(QWidget):

    def __init__(self, showAlpha: bool, showStepLength: bool, parent: Optional[QWidget]) -> None:
        super().__init__(parent)
        self.basicParametersView = TikeBasicParametersView.createInstance(
            showAlpha, showStepLength)
        self.multigridView = TikeMultigridView.createInstance()
        self.positionCorrectionView = TikePositionCorrectionView.createInstance()
        self.probeCorrectionView = TikeProbeCorrectionView.createInstance()
        self.objectCorrectionView = TikeObjectCorrectionView.createInstance()

    @classmethod
    def createInstance(cls,
                       showAlpha: bool,
                       showStepLength: bool,
                       parent: Optional[QWidget] = None) -> TikeParametersView:
        view = cls(showAlpha, showStepLength, parent)

        layout = QVBoxLayout()
        layout.addWidget(view.basicParametersView)
        layout.addWidget(view.multigridView)
        layout.addWidget(view.positionCorrectionView)
        layout.addWidget(view.probeCorrectionView)
        layout.addWidget(view.objectCorrectionView)
        layout.addStretch()
        view.setLayout(layout)

        return view

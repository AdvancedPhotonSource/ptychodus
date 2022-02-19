from __future__ import annotations

from PyQt5.QtWidgets import *


class TikeAdaptiveMomentView(QGroupBox):
    def __init__(self, parent: QWidget) -> None:
        super().__init__(' Use Adaptive Moment', parent)
        self.mdecaySpinBox = QDoubleSpinBox()
        self.vdecaySpinBox = QDoubleSpinBox()

    @classmethod
    def createInstance(cls, parent: QWidget = None) -> TikeAdaptiveMomentView:
        view = cls(parent)

        view.setCheckable(True)
        view.mdecaySpinBox.setToolTip('The proportion of the first moment '
                                      'that is previous first moments.')
        view.vdecaySpinBox.setToolTip('The proportion of the second moment '
                                      'that is previous second moments.')

        layout = QFormLayout()
        layout.addRow('M Decay:', view.mdecaySpinBox)
        layout.addRow('V Decay:', view.vdecaySpinBox)
        view.setLayout(layout)

        return view


class TikePositionCorrectionView(QGroupBox):
    def __init__(self, parent: QWidget) -> None:
        super().__init__('Position Correction', parent)
        self.positionRegularizationCheckBox = QCheckBox('Use Regularization')
        self.adaptiveMomentView = TikeAdaptiveMomentView.createInstance()

    @classmethod
    def createInstance(cls, parent: QWidget = None) -> TikePositionCorrectionView:
        view = cls(parent)

        view.positionRegularizationCheckBox.setToolTip(
            'Whether the positions are constrained to fit a random error plus affine error model.')

        layout = QFormLayout()
        layout.addRow(view.positionRegularizationCheckBox)
        layout.addRow(view.adaptiveMomentView)
        view.setLayout(layout)

        return view


class TikeProbeCorrectionView(QGroupBox):
    def __init__(self, parent: QWidget) -> None:
        super().__init__('Probe Correction', parent)
        self.sparsityConstraintSpinBox = QDoubleSpinBox()
        self.orthogonalityConstraintCheckBox = QCheckBox('Orthogonality Constraint')
        self.centeredIntensityConstraintCheckBox = QCheckBox('Centered Intensity Constraint')
        self.adaptiveMomentView = TikeAdaptiveMomentView.createInstance()

    @classmethod
    def createInstance(cls, parent: QWidget = None) -> TikeProbeCorrectionView:
        view = cls(parent)

        view.sparsityConstraintSpinBox.setToolTip(
            'Forces a maximum proportion of non-zero elements.')
        view.orthogonalityConstraintCheckBox.setToolTip(
            'Forces probes to be orthogonal each iteration.')
        view.centeredIntensityConstraintCheckBox.setToolTip(
            'Forces the probe intensity to be centered.')

        layout = QFormLayout()
        layout.addRow('Sparsity Constraint:', view.sparsityConstraintSpinBox)
        layout.addRow(view.orthogonalityConstraintCheckBox)
        layout.addRow(view.centeredIntensityConstraintCheckBox)
        layout.addRow(view.adaptiveMomentView)
        view.setLayout(layout)

        return view


class TikeObjectCorrectionView(QGroupBox):
    def __init__(self, parent: QWidget) -> None:
        super().__init__('Object Correction', parent)
        self.positivityConstraintSpinBox = QDoubleSpinBox()
        self.smoothnessConstraintSpinBox = QDoubleSpinBox()
        self.adaptiveMomentView = TikeAdaptiveMomentView.createInstance()

    @classmethod
    def createInstance(cls, parent: QWidget = None) -> TikeObjectCorrectionView:
        view = cls(parent)

        layout = QFormLayout()
        layout.addRow('Positivity Constraint:', view.positivityConstraintSpinBox)
        layout.addRow('Smoothness Constraint:', view.smoothnessConstraintSpinBox)
        layout.addRow(view.adaptiveMomentView)
        view.setLayout(layout)

        return view


class TikeIterationOptionsView(QGroupBox):
    def __init__(self, parent: QWidget) -> None:
        super().__init__('Iteration Options', parent)
        self.numBatchSpinBox = QSpinBox()
        self.numIterSpinBox = QSpinBox()
        self.cgIterSpinBox = QSpinBox()
        self.alphaSpinBox = QDoubleSpinBox()
        self.stepLengthSpinBox = QDoubleSpinBox()

    @classmethod
    def createInstance(cls,
                       showCgIter: bool,
                       showAlpha: bool,
                       showStepLength: bool,
                       parent: QWidget = None) -> TikeIterationOptionsView:
        view = cls(parent)

        view.numBatchSpinBox.setToolTip(
            'The dataset is divided into this number of groups where each group is processed sequentially.'
        )
        view.numIterSpinBox.setToolTip('The number of epochs to process before returning.')
        view.cgIterSpinBox.setToolTip(
            'The number of conjugate directions to search for each update.')
        view.alphaSpinBox.setToolTip('RPIE becomes EPIE when this parameter is 1.')
        view.stepLengthSpinBox.setToolTip(
            'Scales the inital search directions before the line search.')

        layout = QFormLayout()
        layout.addRow('Number of Batches:', view.numBatchSpinBox)
        layout.addRow('Number of Iterations:', view.numIterSpinBox)

        if showCgIter:
            layout.addRow('CG Search Directions:', view.cgIterSpinBox)

        if showAlpha:
            layout.addRow('Alpha:', view.alphaSpinBox)

        if showStepLength:
            layout.addRow('Step Length:', view.stepLengthSpinBox)

        view.setLayout(layout)

        return view


class TikeParametersView(QWidget):
    def __init__(self, showCgIter: bool, showAlpha: bool, showStepLength: bool,
                 parent: QWidget) -> None:
        super().__init__(parent)
        self.positionCorrectionView = TikePositionCorrectionView.createInstance()
        self.probeCorrectionView = TikeProbeCorrectionView.createInstance()
        self.objectCorrectionView = TikeObjectCorrectionView.createInstance()
        self.iterationOptionsView = TikeIterationOptionsView.createInstance(
            showCgIter, showAlpha, showStepLength)

    @classmethod
    def createInstance(cls,
                       showCgIter: bool,
                       showAlpha: bool,
                       showStepLength: bool,
                       parent: QWidget = None) -> TikeParametersView:
        view = cls(showCgIter, showAlpha, showStepLength, parent)

        layout = QVBoxLayout()
        layout.addWidget(view.positionCorrectionView)
        layout.addWidget(view.probeCorrectionView)
        layout.addWidget(view.objectCorrectionView)
        layout.addWidget(view.iterationOptionsView)
        layout.addStretch()
        view.setLayout(layout)

        return view

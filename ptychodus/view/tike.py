from __future__ import annotations

from PyQt5.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QSpinBox, QVBoxLayout, QWidget


class TikeBasicParametersView(QGroupBox):
    def __init__(self, parent: QWidget) -> None:
        super().__init__('Basic Parameters', parent)
        self.useMpiCheckBox = QCheckBox('Use MPI')
        self.numGpusSpinBox = QSpinBox()
        self.noiseModelComboBox = QComboBox()
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
                       parent: QWidget = None) -> TikeBasicParametersView:
        view = cls(parent)

        view.useMpiCheckBox.setToolTip('Whether to use MPI or not.')
        view.numGpusSpinBox.setToolTip(
            'The number of GPUs to use. If the number of GPUs is less than the requested number, only workers for the available GPUs are allocated.'
        )
        view.noiseModelComboBox.setToolTip('The noise model to use for the cost function.')
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
        layout.addRow(view.useMpiCheckBox)
        layout.addRow('Number of GPUs:', view.numGpusSpinBox)
        layout.addRow('Noise Model:', view.noiseModelComboBox)
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


class TikeAdaptiveMomentView(QGroupBox):
    def __init__(self, parent: QWidget) -> None:
        super().__init__(' Use Adaptive Moment', parent)
        self.mdecaySpinBox = QDoubleSpinBox()
        self.vdecaySpinBox = QDoubleSpinBox()

    @classmethod
    def createInstance(cls, parent: QWidget = None) -> TikeAdaptiveMomentView:
        view = cls(parent)

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


class TikeParametersView(QWidget):
    def __init__(self, showCgIter: bool, showAlpha: bool, showStepLength: bool,
                 parent: QWidget) -> None:
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
                       parent: QWidget = None) -> TikeParametersView:
        view = cls(showCgIter, showAlpha, showStepLength, parent)

        layout = QVBoxLayout()
        layout.addWidget(view.basicParametersView)
        layout.addWidget(view.positionCorrectionView)
        layout.addWidget(view.probeCorrectionView)
        layout.addWidget(view.objectCorrectionView)
        layout.addStretch()
        view.setLayout(layout)

        return view

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.parametric import BooleanParameter, RealParameter

from PyQt5.QtCore import QRegularExpression
from PyQt5.QtGui import QRegularExpressionValidator
from PyQt5.QtWidgets import QComboBox, QFormLayout, QGroupBox, QWidget

from ...model.tike import (
    TikeMultigridSettings,
    TikeObjectCorrectionSettings,
    TikePositionCorrectionSettings,
    TikeProbeCorrectionSettings,
    TikeSettings,
)
from ..parametric import (
    CheckBoxParameterViewController,
    CheckableGroupBoxParameterViewController,
    ComboBoxParameterViewController,
    DecimalLineEditParameterViewController,
    DecimalSliderParameterViewController,
    LineEditParameterViewController,
    ParameterViewController,
    SpinBoxParameterViewController,
)

__all__ = [
    'TikeMultigridViewController',
    'TikeObjectCorrectionViewController',
    'TikePositionCorrectionViewController',
    'TikeProbeCorrectionViewController',
]


class TikeParametersViewController(ParameterViewController, Observer):
    def __init__(self, settings: TikeSettings, *, showAlpha: bool) -> None:
        super().__init__()
        self._settings = settings
        self._numGpusViewController = LineEditParameterViewController(
            settings.numGpus,
            QRegularExpressionValidator(QRegularExpression('[\\d,]+')),
            tool_tip='The number of GPUs to use. If the number of GPUs is less than the requested number, only workers for the available GPUs are allocated.',
        )
        self._noiseModelViewController = ComboBoxParameterViewController(
            settings.noiseModel,
            settings.getNoiseModels(),
            tool_tip='The noise model to use for the cost function.',
        )
        self._numBatchViewController = SpinBoxParameterViewController(
            settings.numBatch,
            tool_tip='The dataset is divided into this number of groups where each group is processed sequentially.',
        )
        self._batchMethodViewController = ComboBoxParameterViewController(
            settings.batchMethod,
            settings.getBatchMethods(),
            tool_tip='The name of the batch selection method.',
        )
        self._numIterViewController = SpinBoxParameterViewController(
            settings.numIter, tool_tip='The number of epochs to process before returning.'
        )
        self._convergenceWindowViewController = SpinBoxParameterViewController(
            settings.convergenceWindow,
            tool_tip='The number of epochs to consider for convergence monitoring. Set to any value less than 2 to disable.',
        )
        self._alphaViewController = DecimalSliderParameterViewController(
            settings.alpha, tool_tip='RPIE becomes EPIE when this parameter is 1.'
        )
        self._logLevelComboBox = QComboBox()

        for model in settings.getLogLevels():
            self._logLevelComboBox.addItem(model)

        self._logLevelComboBox.textActivated.connect(settings.setLogLevel)

        self._widget = QGroupBox('Tike Parameters')

        layout = QFormLayout()
        layout.addRow('Number of GPUs:', self._numGpusViewController.getWidget())
        layout.addRow('Noise Model:', self._noiseModelViewController.getWidget())
        layout.addRow('Number of Batches:', self._numBatchViewController.getWidget())
        layout.addRow('Batch Method:', self._batchMethodViewController.getWidget())
        layout.addRow('Number of Iterations:', self._numIterViewController.getWidget())
        layout.addRow('Convergence Window:', self._convergenceWindowViewController.getWidget())

        if showAlpha:
            layout.addRow('Alpha:', self._alphaViewController.getWidget())

        layout.addRow('Log Level:', self._logLevelComboBox)
        self._widget.setLayout(layout)

        self._syncModelToView()
        self._settings.addObserver(self)

    def getWidget(self) -> QWidget:
        return self._widget

    def _syncModelToView(self) -> None:
        self._logLevelComboBox.setCurrentText(self._settings.getLogLevel())

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self._syncModelToView()


class TikeMultigridViewController(CheckableGroupBoxParameterViewController):
    def __init__(self, settings: TikeMultigridSettings) -> None:
        super().__init__(settings.useMultigrid, 'Multigrid')
        self._useMultigrid = settings.useMultigrid
        self._numLevelsController = SpinBoxParameterViewController(
            settings.numLevels,
            tool_tip='The number of times to reduce the problem by a factor of two.',
        )

        layout = QFormLayout()
        layout.addRow('Number of Levels:', self._numLevelsController.getWidget())
        self.getWidget().setLayout(layout)


class TikeAdaptiveMomentViewController(CheckableGroupBoxParameterViewController):
    def __init__(
        self, useAdaptiveMoment: BooleanParameter, mdecay: RealParameter, vdecay: RealParameter
    ) -> None:
        super().__init__(useAdaptiveMoment, 'Adaptive Moment')
        self._useAdaptiveMoment = useAdaptiveMoment
        self._mdecayViewController = DecimalSliderParameterViewController(
            mdecay, tool_tip='The proportion of the first moment that is previous first moments.'
        )
        self._vdecayViewController = DecimalSliderParameterViewController(
            vdecay, tool_tip='The proportion of the second moment that is previous second moments.'
        )

        layout = QFormLayout()
        layout.addRow('M Decay:', self._mdecayViewController.getWidget())
        layout.addRow('V Decay:', self._vdecayViewController.getWidget())
        self.getWidget().setLayout(layout)


class TikeObjectCorrectionViewController(CheckableGroupBoxParameterViewController):
    def __init__(self, settings: TikeObjectCorrectionSettings) -> None:
        super().__init__(settings.useObjectCorrection, 'Object Correction')
        self._positivityConstraintViewController = DecimalSliderParameterViewController(
            settings.positivityConstraint
        )
        self._smoothnessConstraintViewController = DecimalSliderParameterViewController(
            settings.smoothnessConstraint
        )
        self._adaptiveMomentViewController = TikeAdaptiveMomentViewController(
            settings.useAdaptiveMoment, settings.mdecay, settings.vdecay
        )
        self._useMagnitudeClippingViewController = CheckBoxParameterViewController(
            settings.useMagnitudeClipping,
            'Magnitude Clipping',
            tool_tip='Forces the object magnitude to be <= 1.',
        )

        layout = QFormLayout()
        layout.addRow(
            'Positivity Constraint:', self._positivityConstraintViewController.getWidget()
        )
        layout.addRow(
            'Smoothness Constraint:', self._smoothnessConstraintViewController.getWidget()
        )
        layout.addRow(self._adaptiveMomentViewController.getWidget())
        layout.addRow(self._useMagnitudeClippingViewController.getWidget())
        self.getWidget().setLayout(layout)


class TikeProbeSupportViewController(CheckableGroupBoxParameterViewController):
    def __init__(self, settings: TikeProbeCorrectionSettings) -> None:
        super().__init__(settings.useFiniteProbeSupport, 'Finite Probe Support')
        self._weightViewController = DecimalLineEditParameterViewController(
            settings.probeSupportWeight, tool_tip='Weight of the finite probe constraint.'
        )
        self._radiusViewController = DecimalSliderParameterViewController(
            settings.probeSupportRadius,
            tool_tip='Radius of probe support as fraction of probe grid.',
        )
        self._degreeViewController = DecimalLineEditParameterViewController(
            settings.probeSupportDegree,
            tool_tip='Degree of the supergaussian defining the probe support.',
        )

        layout = QFormLayout()
        layout.addRow('Weight:', self._weightViewController.getWidget())
        layout.addRow('Radius:', self._radiusViewController.getWidget())
        layout.addRow('Degree:', self._degreeViewController.getWidget())
        self.getWidget().setLayout(layout)


class TikeProbeCorrectionViewController(CheckableGroupBoxParameterViewController):
    def __init__(self, settings: TikeProbeCorrectionSettings) -> None:
        super().__init__(settings.useProbeCorrection, 'Probe Correction')
        self._forceSparsityViewController = DecimalSliderParameterViewController(
            settings.forceSparsity, tool_tip='Forces this proportion of zero elements.'
        )
        self._forceOrthogonalityViewController = CheckBoxParameterViewController(
            settings.forceOrthogonality,
            'Force Orthogonality',
            tool_tip='Forces probes to be orthogonal each iteration.',
        )
        self._forceCenteredIntensityViewController = CheckBoxParameterViewController(
            settings.forceCenteredIntensity,
            'Force Centered Intensity',
            tool_tip='Forces the probe intensity to be centered.',
        )
        self._supportViewController = TikeProbeSupportViewController(settings)
        self._adaptiveMomentViewController = TikeAdaptiveMomentViewController(
            settings.useAdaptiveMoment, settings.mdecay, settings.vdecay
        )
        self._additionalProbePenaltyViewController = DecimalLineEditParameterViewController(
            settings.additionalProbePenalty,
            tool_tip='Penalty applied to the last probe for existing.',
        )

        layout = QFormLayout()
        layout.addRow('Force Sparsity:', self._forceSparsityViewController.getWidget())
        layout.addRow(self._forceOrthogonalityViewController.getWidget())
        layout.addRow(self._forceCenteredIntensityViewController.getWidget())
        layout.addRow(self._supportViewController.getWidget())
        layout.addRow(self._adaptiveMomentViewController.getWidget())
        layout.addRow(
            'Additional Probe Penalty:', self._additionalProbePenaltyViewController.getWidget()
        )
        self.getWidget().setLayout(layout)


class TikePositionCorrectionViewController(CheckableGroupBoxParameterViewController):
    def __init__(self, settings: TikePositionCorrectionSettings) -> None:
        super().__init__(settings.usePositionCorrection, 'Position Correction')
        self._usePositionRegularizationViewController = CheckBoxParameterViewController(
            settings.usePositionRegularization,
            'Use Regularization',
            tool_tip='Whether the positions are constrained to fit a random error plus affine error model.',
        )
        self._adaptiveMomentViewController = TikeAdaptiveMomentViewController(
            settings.useAdaptiveMoment, settings.mdecay, settings.vdecay
        )
        self._updateMagnitudeLimitViewController = DecimalLineEditParameterViewController(
            settings.updateMagnitudeLimit,
            tool_tip='When set to a positive number, x and y update magnitudes are clipped (limited) to this value.',
        )

        layout = QFormLayout()
        layout.addRow(self._usePositionRegularizationViewController.getWidget())
        layout.addRow(self._adaptiveMomentViewController.getWidget())
        layout.addRow(
            'Update Magnitude Limit:', self._updateMagnitudeLimitViewController.getWidget()
        )
        self.getWidget().setLayout(layout)

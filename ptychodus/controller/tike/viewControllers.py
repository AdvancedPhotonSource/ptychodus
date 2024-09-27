from ptychodus.api.observer import Observable, Observer
from ptychodus.api.parametric import BooleanParameter, RealParameter

from PyQt5.QtCore import QRegularExpression
from PyQt5.QtGui import QRegularExpressionValidator
from PyQt5.QtWidgets import QComboBox, QFormLayout, QGroupBox, QWidget

from ...model.tike import (
    TikeMultigridSettings,
    TikeObjectCorrectionSettings,
    TikeProbeCorrectionSettings,
    TikeSettings,
)
from ..parametric import (
    CheckBoxParameterViewController,
    ComboBoxParameterViewController,
    DecimalLineEditParameterViewController,
    DecimalSliderParameterViewController,
    LineEditParameterViewController,
    ParameterViewController,
    SpinBoxParameterViewController,
)

__all__ = [
    "TikeMultigridViewController",
    "TikeObjectCorrectionViewController",
    "TikePositionCorrectionViewController",
    "TikeProbeCorrectionViewController",
]


class TikeParametersViewController(ParameterViewController, Observer):
    def __init__(self, settings: TikeSettings, *, showAlpha: bool, showStepLength: bool) -> None:
        super().__init__()
        self._settings = settings
        self._numGpusViewController = LineEditParameterViewController(
            settings.numGpus,
            QRegularExpressionValidator(QRegularExpression("[\\d,]+")),
            tooltip="The number of GPUs to use. If the number of GPUs is less than the requested number, only workers for the available GPUs are allocated.",
        )
        self._noiseModelViewController = ComboBoxParameterViewController(
            settings.noiseModel,
            settings.getNoiseModels(),
            tooltip="The noise model to use for the cost function.",
        )
        self._numBatchViewController = SpinBoxParameterViewController(
            settings.numBatch,
            tooltip="The dataset is divided into this number of groups where each group is processed sequentially.",
        )
        self._batchMethodViewController = ComboBoxParameterViewController(
            settings.batchMethod,
            settings.getBatchMethods(),
            tooltip="The name of the batch selection method.",
        )
        self._numIterViewController = SpinBoxParameterViewController(
            settings.numIter, tooltip="The number of epochs to process before returning."
        )
        self._convergenceWindowViewController = SpinBoxParameterViewController(
            settings.convergenceWindow,
            tooltip="The number of epochs to consider for convergence monitoring. Set to any value less than 2 to disable.",
        )
        self._alphaViewController = DecimalSliderParameterViewController(
            settings.alpha, tooltip="RPIE becomes EPIE when this parameter is 1."
        )
        self._stepLengthViewController = DecimalSliderParameterViewController(
            settings.stepLength,
            tooltip="Scales the inital search directions before the line search.",
        )

        self._logLevelComboBox = QComboBox()

        for model in settings.getLogLevels():
            self._logLevelComboBox.addItem(model)

        self._logLevelComboBox.textActivated.connect(settings.setLogLevel)

        self._widget = QGroupBox("Tike Parameters")
        self._widget.setCheckable(True)

        layout = QFormLayout()
        layout.addRow("Number of GPUs:", self._numGpusViewController.getWidget())
        layout.addRow("Noise Model:", self._noiseModelViewController.getWidget())
        layout.addRow("Number of Batches:", self._numBatchViewController.getWidget())
        layout.addRow("Batch Method:", self._batchMethodViewController.getWidget())
        layout.addRow("Number of Iterations:", self._numIterViewController.getWidget())
        layout.addRow("Convergence Window:", self._convergenceWindowViewController.getWidget())

        if showAlpha:
            layout.addRow("Alpha:", self._alphaViewController.getWidget())

        if showStepLength:
            layout.addRow("Step Length:", self._stepLengthViewController.getWidget())

        layout.addRow("Log Level:", self._logLevelComboBox)
        self._widget.setLayout(layout)

        self._syncModelToView()

    def _syncModelToView(self) -> None:
        self._logLevelComboBox.setCurrentText(self._settings.getLogLevel())


class TikeMultigridViewController(ParameterViewController, Observer):
    def __init__(self, settings: TikeMultigridSettings) -> None:
        super().__init__()
        self._useMultigrid = settings.useMultigrid
        self._numLevelsController = SpinBoxParameterViewController(
            settings.numLevels,
            tooltip="The number of times to reduce the problem by a factor of two.",
        )
        self._widget = QGroupBox("Multigrid")
        self._widget.setCheckable(True)

        layout = QFormLayout()
        layout.addRow("Number of Levels:", self._numLevelsController.getWidget())
        self._widget.setLayout(layout)

        self._syncModelToView()
        self._widget.toggled.connect(settings.useMultigrid.setValue)
        self._useMultigrid.addObserver(self)

    def getWidget(self) -> QWidget:
        return self._widget

    def _syncModelToView(self) -> None:
        self._widget.setChecked(self._useMultigrid.getValue())

    def update(self, observable: Observable) -> None:
        if observable is self._useMultigrid:
            self._syncModelToView()


class TikeAdaptiveMomentViewController(ParameterViewController, Observer):
    def __init__(
        self, useAdaptiveMoment: BooleanParameter, mdecay: RealParameter, vdecay: RealParameter
    ) -> None:
        super().__init__()
        self._useAdaptiveMoment = useAdaptiveMoment
        self._mdecayViewController = DecimalSliderParameterViewController(
            mdecay, tooltip="The proportion of the first moment that is previous first moments."
        )
        self._vdecayViewController = DecimalSliderParameterViewController(
            vdecay, tooltip="The proportion of the second moment that is previous second moments."
        )
        self._widget = QGroupBox("Adaptive Moment")
        self._widget.setCheckable(True)

        layout = QFormLayout()
        layout.addRow("M Decay:", self._mdecayViewController.getWidget())
        layout.addRow("V Decay:", self._vdecayViewController.getWidget())
        self._widget.setLayout(layout)

        self._syncModelToView()
        self._widget.toggled.connect(useAdaptiveMoment.setValue)
        self._useAdaptiveMoment.addObserver(self)

    def getWidget(self) -> QWidget:
        return self._widget

    def _syncModelToView(self) -> None:
        self._widget.setChecked(self._useAdaptiveMoment.getValue())

    def update(self, observable: Observable) -> None:
        if observable is self._useAdaptiveMoment:
            self._syncModelToView()


class TikeObjectCorrectionViewController(ParameterViewController, Observer):
    def __init__(self, settings: TikeObjectCorrectionSettings) -> None:
        super().__init__()
        self._useObjectCorrection = settings.useObjectCorrection
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
            "Magnitude Clipping",
            tooltip="Forces the object magnitude to be <= 1.",
        )

        self._widget = QGroupBox("Object Correction")
        self._widget.setCheckable(True)

        layout = QFormLayout()
        layout.addRow(
            "Positivity Constraint:", self._positivityConstraintViewController.getWidget()
        )
        layout.addRow(
            "Smoothness Constraint:", self._smoothnessConstraintViewController.getWidget()
        )
        layout.addRow(self._adaptiveMomentViewController.getWidget())
        layout.addRow(self._useMagnitudeClippingViewController.getWidget())
        self._widget.setLayout(layout)

        self._syncModelToView()
        self._widget.toggled.connect(settings.useObjectCorrection.setValue)
        self._useObjectCorrection.addObserver(self)

    def getWidget(self) -> QWidget:
        return self._widget

    def _syncModelToView(self) -> None:
        self._widget.setChecked(self._useObjectCorrection.getValue())

    def update(self, observable: Observable) -> None:
        if observable is self._useObjectCorrection:
            self._syncModelToView()


class TikeProbeSupportViewController(ParameterViewController, Observer):
    def __init__(self, settings: TikeProbeCorrectionSettings) -> None:
        super().__init__()
        self._useFiniteProbeSupport = settings.useFiniteProbeSupport
        self._weightViewController = DecimalLineEditParameterViewController(
            settings.probeSupportWeight, tooltip="Weight of the finite probe constraint."
        )
        self._radiusViewController = DecimalSliderParameterViewController(
            settings.probeSupportRadius,
            tooltip="Radius of probe support as fraction of probe grid.",
        )
        self._degreeViewController = DecimalLineEditParameterViewController(
            settings.probeSupportDegree,
            tooltip="Degree of the supergaussian defining the probe support.",
        )
        self._widget = QGroupBox("Finite Probe Support")
        self._widget.setCheckable(True)

        layout = QFormLayout()
        layout.addRow("Weight:", self._weightViewController.getWidget())
        layout.addRow("Radius:", self._radiusViewController.getWidget())
        layout.addRow("Degree:", self._degreeViewController.getWidget())
        self._widget.setLayout(layout)

        self._syncModelToView()
        self._widget.toggled.connect(settings.useFiniteProbeSupport.setValue)
        self._useFiniteProbeSupport.addObserver(self)

    def getWidget(self) -> QWidget:
        return self._widget

    def _syncModelToView(self) -> None:
        self._widget.setChecked(self._useFiniteProbeSupport.getValue())

    def update(self, observable: Observable) -> None:
        if observable is self._useFiniteProbeSupport:
            self._syncModelToView()


class TikeProbeCorrectionViewController(ParameterViewController, Observer):
    def __init__(self, settings: TikeProbeCorrectionSettings) -> None:
        super().__init__()
        self._useProbeCorrection = settings.useProbeCorrection
        self._forceSparsityViewController = DecimalSliderParameterViewController(
            settings.forceSparsity, tooltip="Forces this proportion of zero elements."
        )
        self._forceOrthogonalityViewController = CheckBoxParameterViewController(
            settings.forceOrthogonality,
            "Force Orthogonality",
            tooltip="Forces probes to be orthogonal each iteration.",
        )
        self._forceCenteredIntensityViewController = CheckBoxParameterViewController(
            settings.forceCenteredIntensity,
            "Force Centered Intensity",
            tooltip="Forces the probe intensity to be centered.",
        )
        self._supportViewController = TikeProbeSupportViewController(settings)
        self._adaptiveMomentViewController = TikeAdaptiveMomentViewController(
            settings.useAdaptiveMoment, settings.mdecay, settings.vdecay
        )
        self._additionalProbePenaltyViewController = DecimalLineEditParameterViewController(
            settings.additionalProbePenalty,
            tooltip="Penalty applied to the last probe for existing.",
        )
        self._widget = QGroupBox("Probe Correction")
        self._widget.setCheckable(True)

        layout = QFormLayout()
        layout.addRow("Force Sparsity:", self._forceSparsityViewController.getWidget())
        layout.addRow(self._forceOrthogonalityViewController.getWidget())
        layout.addRow(self._forceCenteredIntensityViewController.getWidget())
        layout.addRow(self._supportViewController.getWidget())
        layout.addRow(self._adaptiveMomentViewController.getWidget())
        layout.addRow(
            "Additional Probe Penalty:", self._additionalProbePenaltyViewController.getWidget()
        )
        self._widget.setLayout(layout)

        self._syncModelToView()
        self._widget.toggled.connect(settings.useProbeCorrection.setValue)
        self._useProbeCorrection.addObserver(self)

    def getWidget(self) -> QWidget:
        return self._widget

    def _syncModelToView(self) -> None:
        self._widget.setChecked(self._useProbeCorrection.getValue())

    def update(self, observable: Observable) -> None:
        if observable is self._useProbeCorrection:
            self._syncModelToView()


class TikePositionCorrectionViewController(ParameterViewController, Observer):
    def __init__(
        self,
        usePositionCorrection: BooleanParameter,
        usePositionRegularization: BooleanParameter,
        adaptiveMomentViewController: TikeAdaptiveMomentViewController,
        updateMagnitudeLimit: RealParameter,
    ) -> None:
        self._usePositionCorrection = usePositionCorrection
        self._usePositionRegularizationViewController = CheckBoxParameterViewController(
            usePositionRegularization,
            "Use Regularization",
            tooltip="Whether the positions are constrained to fit a random error plus affine error model.",
        )
        self._adaptiveMomentViewController = adaptiveMomentViewController
        self._updateMagnitudeLimitViewController = DecimalLineEditParameterViewController(
            updateMagnitudeLimit,
            tooltip="When set to a positive number, x and y update magnitudes are clipped (limited) to this value.",
        )
        self._widget = QGroupBox("Position Correction")
        self._widget.setCheckable(True)

        layout = QFormLayout()
        layout.addRow(self._usePositionRegularizationViewController.getWidget())
        layout.addRow(self._adaptiveMomentViewController.getWidget())
        layout.addRow(
            "Update Magnitude Limit:", self._updateMagnitudeLimitViewController.getWidget()
        )
        self._widget.setLayout(layout)

        self._syncModelToView()
        self._widget.toggled.connect(usePositionCorrection.setValue)
        self._usePositionCorrection.addObserver(self)

    def getWidget(self) -> QWidget:
        return self._widget

    def _syncModelToView(self) -> None:
        self._widget.setChecked(self._usePositionCorrection.getValue())

    def update(self, observable: Observable) -> None:
        if observable is self._usePositionCorrection:
            self._syncModelToView()

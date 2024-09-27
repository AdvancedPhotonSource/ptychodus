from ptychodus.api.observer import Observable, Observer
from ptychodus.api.parametric import BooleanParameter, IntegerParameter, RealParameter

from PyQt5.QtWidgets import QFormLayout, QGroupBox, QWidget

from ..parametric import (CheckBoxParameterViewController, DecimalLineEditParameterViewController,
                          DecimalSliderParameterViewController, ParameterViewController,
                          SpinBoxParameterViewController)

__all__ = [
    "TikeMultigridViewController",
    "TikeObjectCorrectionViewController",
    "TikePositionCorrectionViewController",
    "TikeProbeCorrectionViewController",
]


class TikeMultigridViewController(ParameterViewController, Observer):

    def __init__(self, useMultigrid: BooleanParameter, numLevels: IntegerParameter) -> None:
        super().__init__()
        self._useMultigrid = useMultigrid
        self._numLevelsController = SpinBoxParameterViewController(
            numLevels, tooltip="The number of times to reduce the problem by a factor of two.")
        self._widget = QGroupBox("Multigrid")
        self._widget.setCheckable(True)

        layout = QFormLayout()
        layout.addRow("Number of Levels:", self._numLevelsController.getWidget())
        self._widget.setLayout(layout)

        self._syncModelToView()
        self._widget.toggled.connect(useMultigrid.setValue)
        self._useMultigrid.addObserver(self)

    def getWidget(self) -> QWidget:
        return self._widget

    def _syncModelToView(self) -> None:
        self._widget.setChecked(self._useMultigrid.getValue())

    def update(self, observable: Observable) -> None:
        if observable is self._useMultigrid:
            self._syncModelToView()


class TikeAdaptiveMomentViewController(ParameterViewController, Observer):

    def __init__(self, useAdaptiveMoment: BooleanParameter, mdecay: RealParameter,
                 vdecay: RealParameter) -> None:
        super().__init__()
        self._useAdaptiveMoment = useAdaptiveMoment
        self._mdecayViewController = DecimalSliderParameterViewController(
            mdecay, tooltip="The proportion of the first moment that is previous first moments.")
        self._vdecayViewController = DecimalSliderParameterViewController(
            vdecay, tooltip="The proportion of the second moment that is previous second moments.")
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

    def __init__(self, useObjectCorrection: BooleanParameter, positivityConstraint: RealParameter,
                 smoothnessConstraint: RealParameter,
                 adaptiveMomentViewController: TikeAdaptiveMomentViewController,
                 useMagnitudeClipping: BooleanParameter) -> None:
        super().__init__()
        self._useObjectCorrection = useObjectCorrection
        self._positivityConstraintViewController = DecimalSliderParameterViewController(
            positivityConstraint)
        self._smoothnessConstraintViewController = DecimalSliderParameterViewController(
            smoothnessConstraint)
        self._adaptiveMomentViewController = adaptiveMomentViewController
        self._useMagnitudeClippingViewController = CheckBoxParameterViewController(
            useMagnitudeClipping,
            "Magnitude Clipping",
            tooltip="Forces the object magnitude to be <= 1.")

        self._widget = QGroupBox("Object Correction")
        self._widget.setCheckable(True)

        layout = QFormLayout()
        layout.addRow("Positivity Constraint:",
                      self._positivityConstraintViewController.getWidget())
        layout.addRow("Smoothness Constraint:",
                      self._smoothnessConstraintViewController.getWidget())
        layout.addRow(self._adaptiveMomentViewController.getWidget())
        layout.addRow(self._useMagnitudeClippingViewController.getWidget())
        self._widget.setLayout(layout)

        self._syncModelToView()
        self._widget.toggled.connect(useObjectCorrection.setValue)
        self._useObjectCorrection.addObserver(self)

    def getWidget(self) -> QWidget:
        return self._widget

    def _syncModelToView(self) -> None:
        self._widget.setChecked(self._useObjectCorrection.getValue())

    def update(self, observable: Observable) -> None:
        if observable is self._useObjectCorrection:
            self._syncModelToView()


class TikeProbeSupportViewController(ParameterViewController, Observer):

    def __init__(self, useFiniteProbeSupport: BooleanParameter, weight: RealParameter,
                 radius: RealParameter, degree: RealParameter) -> None:
        super().__init__()
        self._useFiniteProbeSupport = useFiniteProbeSupport
        self._weightViewController = DecimalLineEditParameterViewController(
            weight, tooltip="Weight of the finite probe constraint.")
        self._radiusViewController = DecimalSliderParameterViewController(
            radius, tooltip="Radius of probe support as fraction of probe grid.")
        self._degreeViewController = DecimalLineEditParameterViewController(
            degree,
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
        self._widget.toggled.connect(useFiniteProbeSupport.setValue)
        self._useFiniteProbeSupport.addObserver(self)

    def getWidget(self) -> QWidget:
        return self._widget

    def _syncModelToView(self) -> None:
        self._widget.setChecked(self._useFiniteProbeSupport.getValue())

    def update(self, observable: Observable) -> None:
        if observable is self._useFiniteProbeSupport:
            self._syncModelToView()


class TikeProbeCorrectionViewController(ParameterViewController, Observer):

    def __init__(self, useProbeCorrection: BooleanParameter, forceSparsity: RealParameter,
                 forceOrthogonality: BooleanParameter, forceCenteredIntensity: BooleanParameter,
                 supportViewController: TikeProbeSupportViewController,
                 adaptiveMomentViewController: TikeAdaptiveMomentViewController,
                 additionalProbePenalty: RealParameter) -> None:
        super().__init__()
        self._useProbeCorrection = useProbeCorrection
        self._forceSparsityViewController = DecimalSliderParameterViewController(
            forceSparsity, tooltip="Forces this proportion of zero elements.")
        self._forceOrthogonalityViewController = CheckBoxParameterViewController(
            forceOrthogonality,
            "Force Orthogonality",
            tooltip="Forces probes to be orthogonal each iteration.")
        self._forceCenteredIntensityViewController = CheckBoxParameterViewController(
            forceCenteredIntensity,
            "Force Centered Intensity",
            tooltip="Forces the probe intensity to be centered.")
        self._supportViewController = supportViewController
        self._adaptiveMomentViewController = adaptiveMomentViewController
        self._additionalProbePenaltyViewController = DecimalLineEditParameterViewController(
            additionalProbePenalty,
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
        layout.addRow("Additional Probe Penalty:",
                      self._additionalProbePenaltyViewController.getWidget())
        self._widget.setLayout(layout)

        self._syncModelToView()
        self._widget.toggled.connect(useProbeCorrection.setValue)
        self._useProbeCorrection.addObserver(self)

    def getWidget(self) -> QWidget:
        return self._widget

    def _syncModelToView(self) -> None:
        self._widget.setChecked(self._useProbeCorrection.getValue())

    def update(self, observable: Observable) -> None:
        if observable is self._useProbeCorrection:
            self._syncModelToView()


class TikePositionCorrectionViewController(ParameterViewController, Observer):

    def __init__(self, usePositionCorrection: BooleanParameter,
                 usePositionRegularization: BooleanParameter,
                 adaptiveMomentViewController: TikeAdaptiveMomentViewController,
                 updateMagnitudeLimit: RealParameter) -> None:
        self._usePositionCorrection = usePositionCorrection
        self._usePositionRegularizationViewController = CheckBoxParameterViewController(
            usePositionRegularization,
            "Use Regularization",
            tooltip=
            "Whether the positions are constrained to fit a random error plus affine error model.")
        self._adaptiveMomentViewController = adaptiveMomentViewController
        self._updateMagnitudeLimitViewController = DecimalLineEditParameterViewController(
            updateMagnitudeLimit,
            tooltip=
            "When set to a positive number, x and y update magnitudes are clipped (limited) to this value."
        )
        self._widget = QGroupBox("Position Correction")
        self._widget.setCheckable(True)

        layout = QFormLayout()
        layout.addRow(self._usePositionRegularizationViewController.getWidget())
        layout.addRow(self._adaptiveMomentViewController.getWidget())
        layout.addRow("Update Magnitude Limit:",
                      self._updateMagnitudeLimitViewController.getWidget())
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

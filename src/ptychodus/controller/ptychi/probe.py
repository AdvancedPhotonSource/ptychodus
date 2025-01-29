from PyQt5.QtWidgets import QFormLayout

from ptychodus.api.parametric import (
    BooleanParameter,
    IntegerParameter,
    RealParameter,
    StringParameter,
)

from ...model.ptychi import PtyChiEnumerators, PtyChiProbeSettings
from ..parametric import (
    CheckableGroupBoxParameterViewController,
    ComboBoxParameterViewController,
    DecimalLineEditParameterViewController,
    DecimalSliderParameterViewController,
)
from .optimizer import PtyChiOptimizationPlanViewController, PtyChiOptimizerParameterViewController

__all__ = ['PtyChiProbeViewController']


class PtyChiConstrainProbePowerViewController(CheckableGroupBoxParameterViewController):
    def __init__(
        self,
        constrainPower: BooleanParameter,
        start: IntegerParameter,
        stop: IntegerParameter,
        stride: IntegerParameter,
        num_epochs: IntegerParameter,
    ) -> None:
        super().__init__(
            constrainPower, 'Constrain Power', tool_tip='Whether to constrain probe power.'
        )
        self._planViewController = PtyChiOptimizationPlanViewController(
            start, stop, stride, num_epochs
        )

        layout = QFormLayout()
        layout.addRow('Plan:', self._planViewController.getWidget())
        self.getWidget().setLayout(layout)


class PtyChiOrthogonalizeIncoherentModesViewController(CheckableGroupBoxParameterViewController):
    def __init__(
        self,
        orthogonalizeModes: BooleanParameter,
        start: IntegerParameter,
        stop: IntegerParameter,
        stride: IntegerParameter,
        method: StringParameter,
        num_epochs: IntegerParameter,
        enumerators: PtyChiEnumerators,
    ) -> None:
        super().__init__(
            orthogonalizeModes,
            'Orthogonalize Incoherent Modes',
            tool_tip='Whether to orthogonalize incoherent probe modes.',
        )
        self._planViewController = PtyChiOptimizationPlanViewController(
            start, stop, stride, num_epochs
        )
        self._methodViewController = ComboBoxParameterViewController(
            method,
            enumerators.orthogonalizationMethods(),
            tool_tip='Method to use for incoherent mode orthogonalization.',
        )

        layout = QFormLayout()
        layout.addRow('Plan:', self._planViewController.getWidget())
        layout.addRow('Method:', self._methodViewController.getWidget())
        self.getWidget().setLayout(layout)


class PtyChiOrthogonalizeOPRModesViewController(CheckableGroupBoxParameterViewController):
    def __init__(
        self,
        orthogonalizeModes: BooleanParameter,
        start: IntegerParameter,
        stop: IntegerParameter,
        stride: IntegerParameter,
        num_epochs: IntegerParameter,
    ) -> None:
        super().__init__(
            orthogonalizeModes,
            'Orthogonalize OPR Modes',
            tool_tip='Whether to orthogonalize OPR modes.',
        )
        self._planViewController = PtyChiOptimizationPlanViewController(
            start, stop, stride, num_epochs
        )

        layout = QFormLayout()
        layout.addRow('Plan:', self._planViewController.getWidget())
        self.getWidget().setLayout(layout)


class PtyChiConstrainSupportViewController(CheckableGroupBoxParameterViewController):
    def __init__(
        self,
        constrainSupport: BooleanParameter,
        start: IntegerParameter,
        stop: IntegerParameter,
        stride: IntegerParameter,
        threshold: RealParameter,
        num_epochs: IntegerParameter,
    ) -> None:
        super().__init__(
            constrainSupport,
            'Constrain Support',
            tool_tip='When enabled, the probe will be shrinkwrapped so that small values are set to zero.',
        )
        self._planViewController = PtyChiOptimizationPlanViewController(
            start, stop, stride, num_epochs
        )
        self._thresholdViewController = DecimalLineEditParameterViewController(
            threshold, tool_tip='Threshold for the probe support constraint.'
        )

        layout = QFormLayout()
        layout.addRow('Plan:', self._planViewController.getWidget())
        layout.addRow('Threshold:', self._thresholdViewController.getWidget())
        self.getWidget().setLayout(layout)


class PtyChiConstrainCenterViewController(CheckableGroupBoxParameterViewController):
    def __init__(
        self,
        constrainCenter: BooleanParameter,
        start: IntegerParameter,
        stop: IntegerParameter,
        stride: IntegerParameter,
        num_epochs: IntegerParameter,
    ) -> None:
        super().__init__(
            constrainCenter,
            'Constrain Center',
            tool_tip='When enabled, the probe center of mass will be constrained to the center of the probe array.',
        )
        self._planViewController = PtyChiOptimizationPlanViewController(
            start, stop, stride, num_epochs
        )

        layout = QFormLayout()
        layout.addRow('Plan:', self._planViewController.getWidget())
        self.getWidget().setLayout(layout)


class PtyChiProbeViewController(CheckableGroupBoxParameterViewController):
    def __init__(
        self,
        settings: PtyChiProbeSettings,
        num_epochs: IntegerParameter,
        enumerators: PtyChiEnumerators,
    ) -> None:
        super().__init__(
            settings.isOptimizable, 'Optimize Probe', tool_tip='Whether the probe is optimizable.'
        )
        self._optimizationPlanViewController = PtyChiOptimizationPlanViewController(
            settings.optimizationPlanStart,
            settings.optimizationPlanStop,
            settings.optimizationPlanStride,
            num_epochs,
        )
        self._optimizerViewController = PtyChiOptimizerParameterViewController(
            settings.optimizer, enumerators
        )
        self._stepSizeViewController = DecimalLineEditParameterViewController(
            settings.stepSize, tool_tip='Optimizer step size'
        )
        self._constrainPowerViewController = PtyChiConstrainProbePowerViewController(
            settings.constrainProbePower,
            settings.constrainProbePowerStart,
            settings.constrainProbePowerStop,
            settings.constrainProbePowerStride,
            num_epochs,
        )
        self._orthogonalizeIncoherentModesViewController = (
            PtyChiOrthogonalizeIncoherentModesViewController(
                settings.orthogonalizeIncoherentModes,
                settings.orthogonalizeIncoherentModesStart,
                settings.orthogonalizeIncoherentModesStop,
                settings.orthogonalizeIncoherentModesStride,
                settings.orthogonalizeIncoherentModesMethod,
                num_epochs,
                enumerators,
            )
        )
        self._orthogonalizeOPRModesViewController = PtyChiOrthogonalizeOPRModesViewController(
            settings.orthogonalizeOPRModes,
            settings.orthogonalizeOPRModesStart,
            settings.orthogonalizeOPRModesStop,
            settings.orthogonalizeOPRModesStride,
            num_epochs,
        )
        self._constrainSupportViewController = PtyChiConstrainSupportViewController(
            settings.constrainSupport,
            settings.constrainSupportStart,
            settings.constrainSupportStop,
            settings.constrainSupportStride,
            settings.constrainSupportThreshold,
            num_epochs,
        )
        self._constrainCenterViewController = PtyChiConstrainCenterViewController(
            settings.constrainCenter,
            settings.constrainCenterStart,
            settings.constrainCenterStop,
            settings.constrainCenterStride,
            num_epochs,
        )
        self._relaxEigenmodeUpdateViewController = DecimalSliderParameterViewController(
            settings.relaxEigenmodeUpdate,
            tool_tip='Relaxation factor for the eigenmode update.',
        )

        layout = QFormLayout()
        layout.addRow('Plan:', self._optimizationPlanViewController.getWidget())
        layout.addRow('Optimizer:', self._optimizerViewController.getWidget())
        layout.addRow('Step Size:', self._stepSizeViewController.getWidget())
        layout.addRow(self._constrainPowerViewController.getWidget())
        layout.addRow(self._orthogonalizeIncoherentModesViewController.getWidget())
        layout.addRow(self._orthogonalizeOPRModesViewController.getWidget())
        layout.addRow(self._constrainSupportViewController.getWidget())
        layout.addRow(self._constrainCenterViewController.getWidget())
        layout.addRow(self._relaxEigenmodeUpdateViewController.getWidget())
        self.getWidget().setLayout(layout)

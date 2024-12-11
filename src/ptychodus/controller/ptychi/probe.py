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
    SpinBoxParameterViewController,
)
from .optimizer import PtyChiOptimizationPlanViewController, PtyChiOptimizerParameterViewController

__all__ = ['PtyChiProbeViewController']


class PtyChiConstrainProbePowerViewController(CheckableGroupBoxParameterViewController):
    def __init__(
        self,
        constrainPower: BooleanParameter,
        stride: IntegerParameter,
    ) -> None:
        super().__init__(
            constrainPower, 'Constrain Power', tool_tip='Whether to constrain probe power.'
        )
        self._strideViewController = SpinBoxParameterViewController(
            stride, tool_tip='Number of epochs between probe power constraint updates.'
        )

        layout = QFormLayout()
        layout.addRow('Stride:', self._strideViewController.getWidget())
        self.getWidget().setLayout(layout)


class PtyChiConstrainSupportViewController(CheckableGroupBoxParameterViewController):
    def __init__(
        self,
        constrainSupport: BooleanParameter,
        threshold: RealParameter,
        stride: IntegerParameter,
    ) -> None:
        super().__init__(
            constrainSupport,
            'Constrain Support',
            tool_tip='When enabled, the probe will be shrinkwrapped so that small values are set to zero.',
        )
        self._thresholdViewController = DecimalLineEditParameterViewController(
            threshold, tool_tip='Threshold for the probe support constraint.'
        )
        self._strideViewController = SpinBoxParameterViewController(
            stride, tool_tip='Number of epochs between probe support constraint updates.'
        )

        layout = QFormLayout()
        layout.addRow('Threshold:', self._thresholdViewController.getWidget())
        layout.addRow('Stride:', self._strideViewController.getWidget())
        self.getWidget().setLayout(layout)


class PtyChiOrthogonalizeIncoherentModesViewController(CheckableGroupBoxParameterViewController):
    def __init__(
        self,
        orthogonalizeModes: BooleanParameter,
        method: StringParameter,
        stride: IntegerParameter,
        enumerators: PtyChiEnumerators,
    ) -> None:
        super().__init__(
            orthogonalizeModes,
            'Orthogonalize Incoherent Modes',
            tool_tip='Whether to orthogonalize incoherent probe modes.',
        )
        self._methodViewController = ComboBoxParameterViewController(
            method,
            enumerators.orthogonalizationMethods(),
            tool_tip='Method to use for incoherent mode orthogonalization.',
        )
        self._strideViewController = SpinBoxParameterViewController(
            stride,
            tool_tip='Number of epochs between orthogonalizing the incoherent probe modes.',
        )

        layout = QFormLayout()
        layout.addRow('Method:', self._methodViewController.getWidget())
        layout.addRow('Stride:', self._strideViewController.getWidget())
        self.getWidget().setLayout(layout)


class PtyChiOrthogonalizeOPRModesViewController(CheckableGroupBoxParameterViewController):
    def __init__(
        self,
        orthogonalizeModes: BooleanParameter,
        stride: IntegerParameter,
    ) -> None:
        super().__init__(
            orthogonalizeModes,
            'Orthogonalize OPR Modes',
            tool_tip='Whether to orthogonalize OPR modes.',
        )
        self._strideViewController = SpinBoxParameterViewController(
            stride, tool_tip='Number of epochs between orthogonalizing the OPR modes.'
        )

        layout = QFormLayout()
        layout.addRow('Stride:', self._strideViewController.getWidget())
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
            settings.constrainProbePower, settings.constrainProbePowerStride
        )
        self._constrainSupportViewController = PtyChiConstrainSupportViewController(
            settings.constrainSupport,
            settings.constrainSupportThreshold,
            settings.constrainSupportStride,
        )
        self._orthogonalizeIncoherentModesViewController = (
            PtyChiOrthogonalizeIncoherentModesViewController(
                settings.orthogonalizeIncoherentModes,
                settings.orthogonalizeIncoherentModesMethod,
                settings.orthogonalizeIncoherentModesStride,
                enumerators,
            )
        )
        self._orthogonalizeOPRModesViewController = PtyChiOrthogonalizeOPRModesViewController(
            settings.orthogonalizeOPRModes,
            settings.orthogonalizeOPRModesStride,
        )

        layout = QFormLayout()
        layout.addRow('Plan:', self._optimizationPlanViewController.getWidget())
        layout.addRow('Optimizer:', self._optimizerViewController.getWidget())
        layout.addRow('Step Size:', self._stepSizeViewController.getWidget())
        layout.addRow(self._constrainPowerViewController.getWidget())
        layout.addRow(self._orthogonalizeIncoherentModesViewController.getWidget())
        layout.addRow(self._orthogonalizeOPRModesViewController.getWidget())
        self.getWidget().setLayout(layout)

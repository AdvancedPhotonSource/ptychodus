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


class ConstrainProbePowerViewController(CheckableGroupBoxParameterViewController):
    def __init__(
        self,
        constrainPower: BooleanParameter,
        power: RealParameter,
        stride: IntegerParameter,
    ) -> None:
        super().__init__(constrainPower, 'Constrain Power')
        self._powerViewController = DecimalLineEditParameterViewController(power)
        self._strideViewController = SpinBoxParameterViewController(stride)

        layout = QFormLayout()
        layout.addRow('Power:', self._powerViewController.getWidget())
        layout.addRow('Stride:', self._strideViewController.getWidget())
        self.getWidget().setLayout(layout)


class OrthogonalizeIncoherentModesViewController(CheckableGroupBoxParameterViewController):
    def __init__(
        self,
        orthogonalizeModes: BooleanParameter,
        method: StringParameter,
        stride: IntegerParameter,
        enumerators: PtyChiEnumerators,
    ) -> None:
        super().__init__(orthogonalizeModes, 'Orthogonalize Incoherent Modes')
        self._methodViewController = ComboBoxParameterViewController(
            method, enumerators.orthogonalizationMethods()
        )
        self._strideViewController = SpinBoxParameterViewController(stride)

        layout = QFormLayout()
        layout.addRow('Method:', self._methodViewController.getWidget())
        layout.addRow('Stride:', self._strideViewController.getWidget())
        self.getWidget().setLayout(layout)


class OrthogonalizeOPRModesViewController(CheckableGroupBoxParameterViewController):
    def __init__(
        self,
        orthogonalizeModes: BooleanParameter,
        stride: IntegerParameter,
    ) -> None:
        super().__init__(orthogonalizeModes, 'Orthogonalize OPR Modes')
        self._orthogonalizeModes = orthogonalizeModes
        self._strideViewController = SpinBoxParameterViewController(stride)

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
        super().__init__(settings.isOptimizable, 'Optimize Probe')
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
        self._constrainPowerViewController = ConstrainProbePowerViewController(
            settings.constrainProbePower, settings.probePower, settings.constrainProbePowerStride
        )
        self._orthogonalizeIncoherentModesViewController = (
            OrthogonalizeIncoherentModesViewController(
                settings.orthogonalizeIncoherentModes,
                settings.orthogonalizeIncoherentModesMethod,
                settings.orthogonalizeIncoherentModesStride,
                enumerators,
            )
        )
        self._orthogonalizeOPRModesViewController = OrthogonalizeOPRModesViewController(
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

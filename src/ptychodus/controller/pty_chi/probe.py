from PyQt5.QtWidgets import QFormLayout, QGroupBox, QWidget

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.parametric import (
    IntegerParameter,
    RealParameter,
    StringParameter,
)

from ...model.pty_chi import PtyChiEnumerators, PtyChiProbeSettings
from ..parametric import (
    ComboBoxParameterViewController,
    DecimalLineEditParameterViewController,
    ParameterViewController,
    SpinBoxParameterViewController,
)
from .optimizer import PtyChiOptimizationPlanViewController, PtyChiOptimizerParameterViewController


class ProbePowerConstraintViewController(ParameterViewController):
    def __init__(
        self,
        power: RealParameter,
        stride: IntegerParameter,
    ) -> None:
        super().__init__()
        self._powerViewController = DecimalLineEditParameterViewController(power)
        self._strideViewController = SpinBoxParameterViewController(stride)
        self._widget = QGroupBox('Power Constraint')
        self._widget.setCheckable(True)  # FIXME

        layout = QFormLayout()
        layout.addRow('Power:', self._powerViewController.getWidget())
        layout.addRow('Stride:', self._strideViewController.getWidget())
        self._widget.setLayout(layout)

    def getWidget(self) -> QWidget:
        return self._widget


class OrthogonalizationViewController(ParameterViewController):
    def __init__(
        self,
        incoherentModesMethod: StringParameter,
        incoherentModesStride: IntegerParameter,
        oprModesStride: IntegerParameter,
        enumerators: PtyChiEnumerators,
    ) -> None:
        super().__init__()
        self._incoherentModesMethodViewController = ComboBoxParameterViewController(
            incoherentModesMethod, enumerators.orthogonalizationMethods()
        )
        self._incoherentModesStrideViewController = SpinBoxParameterViewController(
            incoherentModesStride
        )
        self._oprModesStrideViewController = SpinBoxParameterViewController(oprModesStride)
        self._widget = QGroupBox('Orthogonalization')
        # FIXME enable/disable orthogonalize incoherent/opr separately

        layout = QFormLayout()
        layout.addRow(
            'Incoherent Modes Method:', self._incoherentModesMethodViewController.getWidget()
        )
        layout.addRow(
            'Incoherent Modes Stride:', self._incoherentModesStrideViewController.getWidget()
        )
        layout.addRow('OPR Modes Stride:', self._oprModesStrideViewController.getWidget())
        self._widget.setLayout(layout)

    def getWidget(self) -> QWidget:
        return self._widget


class PtyChiProbeViewController(ParameterViewController, Observer):
    def __init__(
        self,
        settings: PtyChiProbeSettings,
        num_epochs: IntegerParameter,
        enumerators: PtyChiEnumerators,
    ) -> None:
        super().__init__()
        self._isOptimizable = settings.isOptimizable
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
        self._powerConstraintViewController = ProbePowerConstraintViewController(
            settings.probePower, settings.probePowerConstraintStride
        )
        self._orthogonalizationViewController = OrthogonalizationViewController(
            settings.orthogonalizeIncoherentModesMethod,
            settings.orthogonalizeIncoherentModesStride,
            settings.orthogonalizeOPRModesStride,
            enumerators,
        )
        self._widget = QGroupBox('Optimize Probe')
        self._widget.setCheckable(True)

        layout = QFormLayout()
        layout.addRow('Plan:', self._optimizationPlanViewController.getWidget())
        layout.addRow('Optimizer:', self._optimizerViewController.getWidget())
        layout.addRow('Step Size:', self._stepSizeViewController.getWidget())
        layout.addRow(self._powerConstraintViewController.getWidget())
        layout.addRow(self._orthogonalizationViewController.getWidget())
        self._widget.setLayout(layout)

        self._syncModelToView()
        self._widget.toggled.connect(self._isOptimizable.setValue)
        self._isOptimizable.addObserver(self)

    def getWidget(self) -> QWidget:
        return self._widget

    def _syncModelToView(self) -> None:
        self._widget.setChecked(self._isOptimizable.getValue())

    def update(self, observable: Observable) -> None:
        if observable is self._isOptimizable:
            self._syncModelToView()

from PyQt5.QtWidgets import QFormLayout, QGroupBox, QWidget

from ptychodus.api.observer import Observable, Observer

from ...model.pty_chi import PtyChiProbeSettings
from ..parametric import DecimalLineEditParameterViewController, ParameterViewController
from .optimizer import PtyChiOptimizationPlanViewController, PtyChiOptimizerParameterViewController


class PtyChiProbeViewController(ParameterViewController, Observer):
    def __init__(self, settings: PtyChiProbeSettings) -> None:
        super().__init__()
        self._isOptimizable = settings.isOptimizable
        self._optimizationPlanViewController = PtyChiOptimizationPlanViewController(
            settings.optimizationPlanStart,
            settings.optimizationPlanStop,
            settings.optimizationPlanStride,
        )
        self._optimizerViewController = PtyChiOptimizerParameterViewController(settings.optimizer)
        self._stepSizeViewController = DecimalLineEditParameterViewController(
            settings.stepSize, tool_tip='Optimizer step size'
        )
        self._widget = QGroupBox('Optimize Probe')
        self._widget.setCheckable(True)

        layout = QFormLayout()
        layout.addRow('Plan:', self._optimizationPlanViewController.getWidget())
        layout.addRow('Optimizer:', self._optimizerViewController.getWidget())
        layout.addRow('Step Size:', self._stepSizeViewController.getWidget())
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


# FIXME probe_orthogonalize_incoherent_modes_method
# FIXME probe_power
# FIXME probe_power_constraint_stride
# FIXME orthogonalize_incoherent_modes
# FIXME orthogonalize_incoherent_modes_stride
# FIXME orthogonalize_incoherent_modes_method
# FIXME orthogonalize_opr_modes
# FIXME orthogonalize_opr_modes_stride

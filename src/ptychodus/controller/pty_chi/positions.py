from PyQt5.QtWidgets import QFormLayout, QGroupBox, QWidget

from ptychodus.api.observer import Observable, Observer

from ...model.pty_chi import PtyChiEnumerators, PtyChiProbePositionSettings
from ..parametric import DecimalLineEditParameterViewController, ParameterViewController
from .optimizer import PtyChiOptimizationPlanViewController, PtyChiOptimizerParameterViewController


class PtyChiProbePositionsViewController(ParameterViewController, Observer):
    def __init__(
        self, settings: PtyChiProbePositionSettings, enumerators: PtyChiEnumerators
    ) -> None:
        super().__init__()
        self._isOptimizable = settings.isOptimizable
        self._optimizationPlanViewController = PtyChiOptimizationPlanViewController(
            settings.optimizationPlanStart,
            settings.optimizationPlanStop,
            settings.optimizationPlanStride,
        )
        self._optimizerViewController = PtyChiOptimizerParameterViewController(
            settings.optimizer, enumerators
        )
        self._stepSizeViewController = DecimalLineEditParameterViewController(
            settings.stepSize, tool_tip='Optimizer step size'
        )
        self._updateMagnitudeLimitViewController = DecimalLineEditParameterViewController(
            settings.updateMagnitudeLimit,
            tool_tip='When set to a positive number, limit update magnitudes to this value.',
        )
        self._widget = QGroupBox('Optimize Probe Positions')
        self._widget.setCheckable(True)

        layout = QFormLayout()
        layout.addRow('Plan:', self._optimizationPlanViewController.getWidget())
        layout.addRow('Optimizer:', self._optimizerViewController.getWidget())
        layout.addRow('Step Size:', self._stepSizeViewController.getWidget())
        layout.addRow(
            'Update Magnitude Limit:', self._updateMagnitudeLimitViewController.getWidget()
        )
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

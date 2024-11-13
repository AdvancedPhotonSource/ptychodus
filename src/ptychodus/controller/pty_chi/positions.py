from PyQt5.QtWidgets import QFormLayout, QGroupBox, QWidget

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.parametric import IntegerParameter, RealParameter

from ...model.pty_chi import PtyChiEnumerators, PtyChiProbePositionSettings
from ..parametric import (
    CheckBoxParameterViewController,
    ComboBoxParameterViewController,
    DecimalLineEditParameterViewController,
    ParameterViewController,
)
from .optimizer import PtyChiOptimizationPlanViewController, PtyChiOptimizerParameterViewController


class UpdateMagnitudeLimitViewController(ParameterViewController):
    def __init__(
        self,
        updateMagnitudeLimit: RealParameter,
    ) -> None:
        super().__init__()
        self._viewController = DecimalLineEditParameterViewController(
            updateMagnitudeLimit,
            tool_tip='When set to a positive number, limit update magnitudes to this value.',
        )
        self._widget = QGroupBox('Limit Update Magnitude')
        self._widget.setCheckable(True)  # FIXME

        layout = QFormLayout()
        layout.addRow('Limit:', self._viewController.getWidget())
        self._widget.setLayout(layout)

    def getWidget(self) -> QWidget:
        return self._widget


class PtyChiProbePositionsViewController(ParameterViewController, Observer):
    def __init__(
        self,
        settings: PtyChiProbePositionSettings,
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
        self._algorithmViewController = ComboBoxParameterViewController(
            settings.positionCorrectionType, enumerators.positionCorrectionTypes()
        )
        self._updateMagnitudeLimitViewController = UpdateMagnitudeLimitViewController(
            settings.updateMagnitudeLimit,
        )
        self._constrainCentroidViewController = CheckBoxParameterViewController(
            settings.constrainCentroid, 'Constrain Centroid'
        )
        self._widget = QGroupBox('Optimize Probe Positions')
        self._widget.setCheckable(True)

        layout = QFormLayout()
        layout.addRow('Plan:', self._optimizationPlanViewController.getWidget())
        layout.addRow('Optimizer:', self._optimizerViewController.getWidget())
        layout.addRow('Step Size:', self._stepSizeViewController.getWidget())
        layout.addRow('Algorithm:', self._algorithmViewController.getWidget())
        layout.addRow(self._updateMagnitudeLimitViewController.getWidget())
        layout.addRow(self._constrainCentroidViewController.getWidget())
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

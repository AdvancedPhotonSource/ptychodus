from PyQt5.QtWidgets import QFormLayout, QGroupBox, QWidget

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.parametric import IntegerParameter

from ...model.pty_chi import PtyChiEnumerators, PtyChiOPRSettings
from ..parametric import (
    CheckBoxParameterViewController,
    DecimalLineEditParameterViewController,
    ParameterViewController,
)
from .optimizer import PtyChiOptimizationPlanViewController, PtyChiOptimizerParameterViewController


class PtyChiOPRViewController(ParameterViewController, Observer):
    def __init__(
        self,
        settings: PtyChiOPRSettings,
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
        self._optimizeIntensitiesViewController = CheckBoxParameterViewController(
            settings.optimizeIntensities,
            'Optimize Intensities',
        )
        self._optimizeEigenmodeWeightsViewController = CheckBoxParameterViewController(
            settings.optimizeEigenmodeWeights,
            'Optimize Eigenmode Weights',
        )
        self._widget = QGroupBox('Orthogonal Probe Relaxation')
        self._widget.setCheckable(True)

        layout = QFormLayout()
        layout.addRow('Plan:', self._optimizationPlanViewController.getWidget())
        layout.addRow('Optimizer:', self._optimizerViewController.getWidget())
        layout.addRow('Step Size:', self._stepSizeViewController.getWidget())
        layout.addRow(self._optimizeIntensitiesViewController.getWidget())
        layout.addRow(self._optimizeEigenmodeWeightsViewController.getWidget())
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

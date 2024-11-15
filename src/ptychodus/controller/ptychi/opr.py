from PyQt5.QtWidgets import QFormLayout

from ptychodus.api.parametric import IntegerParameter

from ...model.ptychi import PtyChiEnumerators, PtyChiOPRSettings
from ..parametric import (
    CheckBoxParameterViewController,
    CheckableGroupBoxParameterViewController,
    DecimalLineEditParameterViewController,
)
from .optimizer import PtyChiOptimizationPlanViewController, PtyChiOptimizerParameterViewController


class PtyChiOPRViewController(CheckableGroupBoxParameterViewController):
    def __init__(
        self,
        settings: PtyChiOPRSettings,
        num_epochs: IntegerParameter,
        enumerators: PtyChiEnumerators,
    ) -> None:
        super().__init__(settings.isOptimizable, 'Orthogonal Probe Relaxation')
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

        layout = QFormLayout()
        layout.addRow('Plan:', self._optimizationPlanViewController.getWidget())
        layout.addRow('Optimizer:', self._optimizerViewController.getWidget())
        layout.addRow('Step Size:', self._stepSizeViewController.getWidget())
        layout.addRow(self._optimizeIntensitiesViewController.getWidget())
        layout.addRow(self._optimizeEigenmodeWeightsViewController.getWidget())
        self.getWidget().setLayout(layout)

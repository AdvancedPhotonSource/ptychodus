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
        super().__init__(
            settings.isOptimizable,
            'Orthogonal Probe Relaxation',
            tool_tip='Whether OPR modes are optimizable.',
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
        self._optimizeIntensitiesViewController = CheckBoxParameterViewController(
            settings.optimizeIntensities,
            'Optimize Intensities',
            tool_tip='Whether to optimize intensity variation (i.e., the weight of the first OPR mode).',
        )
        self._optimizeEigenmodeWeightsViewController = CheckBoxParameterViewController(
            settings.optimizeEigenmodeWeights,
            'Optimize Eigenmode Weights',
            tool_tip='Whether to optimize eigenmode weights (i.e., the weights of the second and following OPR modes).',
        )

        layout = QFormLayout()
        layout.addRow('Plan:', self._optimizationPlanViewController.getWidget())
        layout.addRow('Optimizer:', self._optimizerViewController.getWidget())
        layout.addRow('Step Size:', self._stepSizeViewController.getWidget())
        layout.addRow(self._optimizeIntensitiesViewController.getWidget())
        layout.addRow(self._optimizeEigenmodeWeightsViewController.getWidget())
        self.getWidget().setLayout(layout)

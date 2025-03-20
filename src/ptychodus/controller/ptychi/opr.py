from PyQt5.QtWidgets import QFormLayout

from ptychodus.api.parametric import BooleanParameter, IntegerParameter, StringParameter

from ...model.ptychi import PtyChiEnumerators, PtyChiOPRSettings
from ..parametric import (
    CheckBoxParameterViewController,
    CheckableGroupBoxParameterViewController,
    ComboBoxParameterViewController,
    DecimalLineEditParameterViewController,
    DecimalSliderParameterViewController,
    SpinBoxParameterViewController,
)
from .optimizer import PtyChiOptimizationPlanViewController, PtyChiOptimizerParameterViewController


class PtyChiSmoothOPRModeWeightsViewController(CheckableGroupBoxParameterViewController):
    def __init__(
        self,
        smoothModeWeights: BooleanParameter,
        start: IntegerParameter,
        stop: IntegerParameter,
        stride: IntegerParameter,
        smoothingMethod: StringParameter,
        polynomialSmoothingDegree: IntegerParameter,
        num_epochs: IntegerParameter,
        enumerators: PtyChiEnumerators,
    ) -> None:
        super().__init__(
            smoothModeWeights,
            'Smooth OPR Mode Weights',
            tool_tip='Smooth the OPR mode weights.',
        )
        self._planViewController = PtyChiOptimizationPlanViewController(
            start, stop, stride, num_epochs
        )
        self._smoothingMethodViewController = ComboBoxParameterViewController(
            smoothingMethod,
            enumerators.oprWeightSmoothingMethods(),
            tool_tip='The method for smoothing OPR mode weights.',
        )
        self._polynomialSmoothingDegreeViewController = SpinBoxParameterViewController(
            polynomialSmoothingDegree,
            tool_tip='The degree of the polynomial used for smoothing OPR mode weights.',
        )

        layout = QFormLayout()
        layout.addRow('Plan:', self._planViewController.get_widget())
        layout.addRow('Smoothing Method:', self._smoothingMethodViewController.get_widget())
        layout.addRow(
            'Polynomial Degree:', self._polynomialSmoothingDegreeViewController.get_widget()
        )
        self.get_widget().setLayout(layout)


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
        self._smoothModeWeightsViewController = PtyChiSmoothOPRModeWeightsViewController(
            settings.smoothModeWeights,
            settings.smoothModeWeightsStart,
            settings.smoothModeWeightsStop,
            settings.smoothModeWeightsStride,
            settings.smoothingMethod,
            settings.polynomialSmoothingDegree,
            num_epochs,
            enumerators,
        )
        self._relaxUpdateViewController = DecimalSliderParameterViewController(
            settings.relaxUpdate,
            tool_tip='Whether to relax the update of the OPR mode weights.',
        )

        layout = QFormLayout()
        layout.addRow('Plan:', self._optimizationPlanViewController.get_widget())
        layout.addRow('Optimizer:', self._optimizerViewController.get_widget())
        layout.addRow('Step Size:', self._stepSizeViewController.get_widget())
        layout.addRow(self._optimizeIntensitiesViewController.get_widget())
        layout.addRow(self._optimizeEigenmodeWeightsViewController.get_widget())
        layout.addRow(self._smoothModeWeightsViewController.get_widget())
        self.get_widget().setLayout(layout)

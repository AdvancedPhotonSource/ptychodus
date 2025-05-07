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
        smooth_mode_weights: BooleanParameter,
        start: IntegerParameter,
        stop: IntegerParameter,
        stride: IntegerParameter,
        smoothing_method: StringParameter,
        polynomial_smoothing_degree: IntegerParameter,
        num_epochs: IntegerParameter,
        enumerators: PtyChiEnumerators,
    ) -> None:
        super().__init__(
            smooth_mode_weights,
            'Smooth OPR Mode Weights',
            tool_tip='Smooth the OPR mode weights',
        )
        self._plan_view_controller = PtyChiOptimizationPlanViewController(
            start, stop, stride, num_epochs
        )
        self._smoothing_method_view_controller = ComboBoxParameterViewController(
            smoothing_method,
            enumerators.opr_weight_smoothing_methods(),
            tool_tip='Method for smoothing OPR mode weights',
        )
        self._polynomial_smoothing_degree_view_controller = SpinBoxParameterViewController(
            polynomial_smoothing_degree,
            tool_tip='Degree of the polynomial used for smoothing OPR mode weights',
        )

        layout = QFormLayout()
        layout.addRow('Plan:', self._plan_view_controller.get_widget())
        layout.addRow('Smoothing Method:', self._smoothing_method_view_controller.get_widget())
        layout.addRow(
            'Polynomial Degree:', self._polynomial_smoothing_degree_view_controller.get_widget()
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
            settings.is_optimizable,
            'Orthogonal Probe Relaxation',
            tool_tip='Whether OPR modes are optimizable',
        )
        self._optimization_plan_view_controller = PtyChiOptimizationPlanViewController(
            settings.optimization_plan_start,
            settings.optimization_plan_stop,
            settings.optimization_plan_stride,
            num_epochs,
        )
        self._optimizer_view_controller = PtyChiOptimizerParameterViewController(
            settings.optimizer, enumerators
        )
        self._step_size_view_controller = DecimalLineEditParameterViewController(
            settings.step_size, tool_tip='Optimizer step size'
        )
        self._optimize_eigenmode_weights_view_controller = CheckBoxParameterViewController(
            settings.optimize_eigenmode_weights,
            'Optimize Eigenmode Weights',
            tool_tip='Whether to optimize eigenmode weights (i.e., the weights of the second and following OPR modes)',
        )
        self._optimize_intensities_view_controller = CheckBoxParameterViewController(
            settings.optimize_intensities,
            'Optimize Intensities',
            tool_tip='Whether to optimize intensity variation (i.e., the weight of the first OPR mode)',
        )
        self._smooth_mode_weights_view_controller = PtyChiSmoothOPRModeWeightsViewController(
            settings.smooth_mode_weights,
            settings.smooth_mode_weights_start,
            settings.smooth_mode_weights_stop,
            settings.smooth_mode_weights_stride,
            settings.smoothing_method,
            settings.polynomial_smoothing_degree,
            num_epochs,
            enumerators,
        )
        self._relax_update_view_controller = DecimalSliderParameterViewController(
            settings.relax_update,
            tool_tip='Whether to relax the update of the OPR mode weights',
        )

        layout = QFormLayout()
        layout.addRow('Plan:', self._optimization_plan_view_controller.get_widget())
        layout.addRow('Optimizer:', self._optimizer_view_controller.get_widget())
        layout.addRow('Step Size:', self._step_size_view_controller.get_widget())
        layout.addRow(self._optimize_intensities_view_controller.get_widget())
        layout.addRow(self._optimize_eigenmode_weights_view_controller.get_widget())
        layout.addRow(self._smooth_mode_weights_view_controller.get_widget())
        layout.addRow('Relax Update:', self._relax_update_view_controller.get_widget())
        self.get_widget().setLayout(layout)

from PyQt5.QtWidgets import QFormLayout

from ptychodus.api.parametric import (
    BooleanParameter,
    IntegerParameter,
    RealParameter,
    StringParameter,
)

from ...model.ptychi import (
    PtyChiDMSettings,
    PtyChiEnumerators,
    PtyChiLSQMLSettings,
    PtyChiObjectSettings,
    PtyChiPIESettings,
)
from ..parametric import (
    CheckBoxParameterViewController,
    CheckableGroupBoxParameterViewController,
    ComboBoxParameterViewController,
    DecimalLineEditParameterViewController,
    DecimalSliderParameterViewController,
    LengthWidgetParameterViewController,
    SpinBoxParameterViewController,
)
from .optimizer import PtyChiOptimizationPlanViewController, PtyChiOptimizerParameterViewController

__all__ = ['PtyChiObjectViewController']


class PtyChiOptimizeSliceSpacingViewController(CheckableGroupBoxParameterViewController):
    def __init__(
        self,
        optimize_slice_spacing: BooleanParameter,
        start: IntegerParameter,
        stop: IntegerParameter,
        stride: IntegerParameter,
        optimizer: StringParameter,
        step_size: RealParameter,
        num_epochs: IntegerParameter,
        enumerators: PtyChiEnumerators,
    ) -> None:
        super().__init__(
            optimize_slice_spacing,
            'Optimize Slice Spacing',
            tool_tip='Whether to optimize the slice spacing',
        )
        self._plan_view_controller = PtyChiOptimizationPlanViewController(
            start,
            stop,
            stride,
            num_epochs,
        )
        self._optimizer_view_controller = PtyChiOptimizerParameterViewController(
            optimizer, enumerators
        )
        self._step_size_view_controller = DecimalLineEditParameterViewController(
            step_size, tool_tip='Optimizer step size'
        )

        layout = QFormLayout()
        layout.addRow('Plan:', self._plan_view_controller.get_widget())
        layout.addRow('Optimizer:', self._optimizer_view_controller.get_widget())
        layout.addRow('Step Size:', self._step_size_view_controller.get_widget())
        self.get_widget().setLayout(layout)


class PtyChiConstrainL1NormViewController(CheckableGroupBoxParameterViewController):
    def __init__(
        self,
        constrain_l1_norm: BooleanParameter,
        start: IntegerParameter,
        stop: IntegerParameter,
        stride: IntegerParameter,
        weight: RealParameter,
        num_epochs: IntegerParameter,
    ) -> None:
        super().__init__(
            constrain_l1_norm,
            'Constrain L\u2081 Norm',
            tool_tip='Whether to constrain the L\u2081 norm',
        )
        self._plan_view_controller = PtyChiOptimizationPlanViewController(
            start, stop, stride, num_epochs
        )
        self._weight_view_controller = DecimalLineEditParameterViewController(
            weight,
            tool_tip='Weight of the L\u2081 norm constraint',
        )

        layout = QFormLayout()
        layout.addRow('Plan:', self._plan_view_controller.get_widget())
        layout.addRow('Weight:', self._weight_view_controller.get_widget())
        self.get_widget().setLayout(layout)


class PtyChiConstrainL2NormViewController(CheckableGroupBoxParameterViewController):
    def __init__(
        self,
        constrain_l2_norm: BooleanParameter,
        start: IntegerParameter,
        stop: IntegerParameter,
        stride: IntegerParameter,
        weight: RealParameter,
        num_epochs: IntegerParameter,
    ) -> None:
        super().__init__(
            constrain_l2_norm,
            'Constrain L\u2082 Norm',
            tool_tip='Whether to constrain the L\u2082 norm',
        )
        self._plan_view_controller = PtyChiOptimizationPlanViewController(
            start, stop, stride, num_epochs
        )
        self._weight_view_controller = DecimalLineEditParameterViewController(
            weight,
            tool_tip='Weight of the L\u2082 norm constraint',
        )

        layout = QFormLayout()
        layout.addRow('Plan:', self._plan_view_controller.get_widget())
        layout.addRow('Weight:', self._weight_view_controller.get_widget())
        self.get_widget().setLayout(layout)


class PtyChiConstrainSmoothnessViewController(CheckableGroupBoxParameterViewController):
    def __init__(
        self,
        constrain_smoothness: BooleanParameter,
        start: IntegerParameter,
        stop: IntegerParameter,
        stride: IntegerParameter,
        alpha: RealParameter,
        num_epochs: IntegerParameter,
    ) -> None:
        super().__init__(
            constrain_smoothness,
            'Constrain Smoothness',
            tool_tip='Whether to constrain smoothness in the magnitude (but not phase) of the object',
        )
        self._plan_view_controller = PtyChiOptimizationPlanViewController(
            start, stop, stride, num_epochs
        )
        self._alpha_view_controller = DecimalSliderParameterViewController(
            alpha, tool_tip='Relaxation smoothing constant'
        )

        layout = QFormLayout()
        layout.addRow('Plan:', self._plan_view_controller.get_widget())
        layout.addRow('Alpha:', self._alpha_view_controller.get_widget())
        self.get_widget().setLayout(layout)


class PtyChiConstrainTotalVariationViewController(CheckableGroupBoxParameterViewController):
    def __init__(
        self,
        constrain_total_variation: BooleanParameter,
        start: IntegerParameter,
        stop: IntegerParameter,
        stride: IntegerParameter,
        weight: RealParameter,
        num_epochs: IntegerParameter,
    ) -> None:
        super().__init__(
            constrain_total_variation,
            'Constrain Total Variation',
            tool_tip='Whether to constrain the total variation',
        )
        self._plan_view_controller = PtyChiOptimizationPlanViewController(
            start, stop, stride, num_epochs
        )
        self._weight_view_controller = DecimalLineEditParameterViewController(
            weight,
            tool_tip='Weight of the total variation constraint',
        )

        layout = QFormLayout()
        layout.addRow('Plan:', self._plan_view_controller.get_widget())
        layout.addRow('Weight:', self._weight_view_controller.get_widget())
        self.get_widget().setLayout(layout)


class PtyChiRemoveGridArtifactsViewController(CheckableGroupBoxParameterViewController):
    def __init__(
        self,
        remove_grid_artifacts: BooleanParameter,
        start: IntegerParameter,
        stop: IntegerParameter,
        stride: IntegerParameter,
        period_x_m: RealParameter,
        period_y_m: RealParameter,
        window_size_px: IntegerParameter,
        direction: StringParameter,
        num_epochs: IntegerParameter,
        enumerators: PtyChiEnumerators,
    ) -> None:
        super().__init__(
            remove_grid_artifacts,
            'Remove Grid Artifacts',
            tool_tip="Whether to remove grid artifacts in the object's phase at the end of an epoch.",
        )
        self._plan_view_controller = PtyChiOptimizationPlanViewController(
            start, stop, stride, num_epochs
        )
        self._period_x_view_controller = LengthWidgetParameterViewController(
            period_x_m, tool_tip='Horizontal period of grid artifacts in meters'
        )
        self._period_y_view_controller = LengthWidgetParameterViewController(
            period_y_m, tool_tip='Vertical period of grid artifacts in meters'
        )
        self._window_size_view_controller = SpinBoxParameterViewController(
            window_size_px, tool_tip='Window size for grid artifact removal in pixels'
        )
        self._direction_view_controller = ComboBoxParameterViewController(
            direction, enumerators.directions(), tool_tip='Direction of grid artifact removal'
        )

        layout = QFormLayout()
        layout.addRow('Plan:', self._plan_view_controller.get_widget())
        layout.addRow('Period X:', self._period_x_view_controller.get_widget())
        layout.addRow('Period Y:', self._period_y_view_controller.get_widget())
        layout.addRow('Window Size [px]:', self._window_size_view_controller.get_widget())
        layout.addRow('Direction:', self._direction_view_controller.get_widget())
        self.get_widget().setLayout(layout)


class PtyChiRegularizeMultisliceViewController(CheckableGroupBoxParameterViewController):
    def __init__(
        self,
        regularize_multislice: BooleanParameter,
        start: IntegerParameter,
        stop: IntegerParameter,
        stride: IntegerParameter,
        weight: RealParameter,
        unwrap_phase: BooleanParameter,
        gradient_method: StringParameter,
        integration_method: StringParameter,
        num_epochs: IntegerParameter,
        enumerators: PtyChiEnumerators,
    ) -> None:
        super().__init__(
            regularize_multislice,
            'Regularize Multislice',
            tool_tip='Whether to regularize multislice objects using cross-slice smoothing',
        )
        self._plan_view_controller = PtyChiOptimizationPlanViewController(
            start, stop, stride, num_epochs
        )
        self._weight_view_controller = DecimalLineEditParameterViewController(
            weight,
            tool_tip='Weight for multislice regularization',
        )
        self._unwrap_phase_view_controller = CheckBoxParameterViewController(
            unwrap_phase,
            'Unwrap Phase',
            tool_tip='Whether to unwrap the phase of the object during multislice regularization',
        )
        self._gradient_method_view_controller = ComboBoxParameterViewController(
            gradient_method,
            enumerators.image_gradient_methods(),
            tool_tip='Method for calculating the phase gradient during phase unwrapping',
        )
        self._integration_method_view_controller = ComboBoxParameterViewController(
            integration_method,
            enumerators.image_integration_methods(),
            tool_tip='Method for integrating the phase gradient during phase unwrapping',
        )

        layout = QFormLayout()
        layout.addRow('Plan:', self._plan_view_controller.get_widget())
        layout.addRow('Weight:', self._weight_view_controller.get_widget())
        layout.addRow(self._unwrap_phase_view_controller.get_widget())
        layout.addRow('Gradient Method:', self._gradient_method_view_controller.get_widget())
        layout.addRow('Integration Method:', self._integration_method_view_controller.get_widget())
        self.get_widget().setLayout(layout)


class PtyChiRemoveObjectProbeAmbiguityViewController(CheckableGroupBoxParameterViewController):
    def __init__(
        self,
        remove_object_probe_ambiguity: BooleanParameter,
        start: IntegerParameter,
        stop: IntegerParameter,
        stride: IntegerParameter,
        num_epochs: IntegerParameter,
    ) -> None:
        super().__init__(
            remove_object_probe_ambiguity,
            'Remove Object Probe Ambiguity',
            tool_tip='Whether to remove object-probe ambiguity',
        )
        self._plan_view_controller = PtyChiOptimizationPlanViewController(
            start, stop, stride, num_epochs
        )

        layout = QFormLayout()
        layout.addRow('Plan:', self._plan_view_controller.get_widget())
        self.get_widget().setLayout(layout)


class PtyChiObjectViewController(CheckableGroupBoxParameterViewController):
    def __init__(
        self,
        settings: PtyChiObjectSettings,
        dm_settings: PtyChiDMSettings | None,
        lsqml_settings: PtyChiLSQMLSettings | None,
        pie_settings: PtyChiPIESettings | None,
        num_epochs: IntegerParameter,
        enumerators: PtyChiEnumerators,
    ) -> None:
        super().__init__(
            settings.is_optimizable,
            'Optimize Object',
            tool_tip='Whether the object is optimizable',
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
        self._optimize_slice_spacing_view_controller = PtyChiOptimizeSliceSpacingViewController(
            settings.optimize_slice_spacing,
            settings.optimize_slice_spacing_start,
            settings.optimize_slice_spacing_stop,
            settings.optimize_slice_spacing_stride,
            settings.optimize_slice_spacing_optimizer,
            settings.optimize_slice_spacing_step_size,
            num_epochs,
            enumerators,
        )
        self._constrain_l1_norm_view_controller = PtyChiConstrainL1NormViewController(
            settings.constrain_l1_norm,
            settings.constrain_l1_norm_start,
            settings.constrain_l1_norm_stop,
            settings.constrain_l1_norm_stride,
            settings.constrain_l1_norm_weight,
            num_epochs,
        )
        self._constrain_l2_norm_view_controller = PtyChiConstrainL2NormViewController(
            settings.constrain_l2_norm,
            settings.constrain_l2_norm_start,
            settings.constrain_l2_norm_stop,
            settings.constrain_l2_norm_stride,
            settings.constrain_l2_norm_weight,
            num_epochs,
        )
        self._constrain_smoothness_view_controller = PtyChiConstrainSmoothnessViewController(
            settings.constrain_smoothness,
            settings.constrain_smoothness_start,
            settings.constrain_smoothness_stop,
            settings.constrain_smoothness_stride,
            settings.constrain_smoothness_alpha,
            num_epochs,
        )
        self._constrain_total_variation_view_controller = (
            PtyChiConstrainTotalVariationViewController(
                settings.constrain_total_variation,
                settings.constrain_total_variation_start,
                settings.constrain_total_variation_stop,
                settings.constrain_total_variation_stride,
                settings.constrain_total_variation_weight,
                num_epochs,
            )
        )
        self._remove_grid_artifacts_view_controller = PtyChiRemoveGridArtifactsViewController(
            settings.remove_grid_artifacts,
            settings.remove_grid_artifacts_start,
            settings.remove_grid_artifacts_stop,
            settings.remove_grid_artifacts_stride,
            settings.remove_grid_artifacts_period_x_m,
            settings.remove_grid_artifacts_period_y_m,
            settings.remove_grid_artifacts_window_size_px,
            settings.remove_grid_artifacts_direction,
            num_epochs,
            enumerators,
        )
        self._regularize_multislice_view_controller = PtyChiRegularizeMultisliceViewController(
            settings.regularize_multislice,
            settings.regularize_multislice_start,
            settings.regularize_multislice_stop,
            settings.regularize_multislice_stride,
            settings.regularize_multislice_weight,
            settings.regularize_multislice_unwrap_phase,
            settings.regularize_multislice_unwrap_phase_image_gradient_method,
            settings.regularize_multislice_unwrap_phase_image_integration_method,
            num_epochs,
            enumerators,
        )
        self._patch_interpolator_view_controller = ComboBoxParameterViewController(
            settings.patch_interpolator,
            enumerators.patch_interpolation_methods(),
            tool_tip='Interpolation method used for extracting and updating patches of the object',
        )
        self._remove_object_probe_ambiguity_view_controller = (
            PtyChiRemoveObjectProbeAmbiguityViewController(
                settings.remove_object_probe_ambiguity,
                settings.remove_object_probe_ambiguity_start,
                settings.remove_object_probe_ambiguity_stop,
                settings.remove_object_probe_ambiguity_stride,
                num_epochs,
            )
        )
        self._build_preconditioner_with_all_modes_view_controller = CheckBoxParameterViewController(
            settings.build_preconditioner_with_all_modes,
            'Build Preconditioner with All Modes',
            tool_tip='Whether to build the preconditioner using all modes',
        )

        layout = QFormLayout()
        layout.addRow('Plan:', self._optimization_plan_view_controller.get_widget())
        layout.addRow('Optimizer:', self._optimizer_view_controller.get_widget())
        layout.addRow('Step Size:', self._step_size_view_controller.get_widget())
        layout.addRow(self._optimize_slice_spacing_view_controller.get_widget())
        layout.addRow(self._constrain_l1_norm_view_controller.get_widget())
        layout.addRow(self._constrain_l2_norm_view_controller.get_widget())
        layout.addRow(self._constrain_smoothness_view_controller.get_widget())
        layout.addRow(self._constrain_total_variation_view_controller.get_widget())
        layout.addRow(self._remove_grid_artifacts_view_controller.get_widget())
        layout.addRow(self._regularize_multislice_view_controller.get_widget())
        layout.addRow('Patch Interpolator:', self._patch_interpolator_view_controller.get_widget())
        layout.addRow(self._remove_object_probe_ambiguity_view_controller.get_widget())
        layout.addRow(self._build_preconditioner_with_all_modes_view_controller.get_widget())

        if dm_settings is not None:
            self._amplitude_clamp_limit_view_controller = DecimalLineEditParameterViewController(
                dm_settings.object_amplitude_clamp_limit,
                tool_tip='Maximum amplitude value for the object',
            )
            layout.addRow(
                'Amplitude Clamp Limit:', self._amplitude_clamp_limit_view_controller.get_widget()
            )

            self._inertia_view_controller = DecimalLineEditParameterViewController(
                dm_settings.object_inertia,
                tool_tip='Inertia for the object update',
            )
            layout.addRow('Inertia:', self._inertia_view_controller.get_widget())

        if lsqml_settings is not None:
            self._object_optimal_step_size_scaler_view_controller = (
                DecimalLineEditParameterViewController(
                    lsqml_settings.object_optimal_step_size_scaler,
                    tool_tip='Optimal step size scaler for the object update',
                )
            )
            layout.addRow(
                'Optimal Step Size Scaler:',
                self._object_optimal_step_size_scaler_view_controller.get_widget(),
            )

            self._object_multimodal_update_view_controller = CheckBoxParameterViewController(
                lsqml_settings.object_multimodal_update,
                'Multimodal Update',
                tool_tip='When checked, the object update direction is calculated and summed over all probe modes rather than only the first mode',
            )
            layout.addRow(self._object_multimodal_update_view_controller.get_widget())

        if pie_settings is not None:
            self._alpha_view_controller = DecimalSliderParameterViewController(
                pie_settings.object_alpha,
                tool_tip='Relaxation factor for the object update',
            )
            layout.addRow('Alpha:', self._alpha_view_controller.get_widget())

        self.get_widget().setLayout(layout)

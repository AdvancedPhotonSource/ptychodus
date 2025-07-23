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
    PtyChiPIESettings,
    PtyChiProbeSettings,
)
from ..parametric import (
    CheckableGroupBoxParameterViewController,
    ComboBoxParameterViewController,
    DecimalLineEditParameterViewController,
    DecimalSliderParameterViewController,
)
from .optimizer import PtyChiOptimizationPlanViewController, PtyChiOptimizerParameterViewController

__all__ = ['PtyChiProbeViewController']


class PtyChiConstrainProbePowerViewController(CheckableGroupBoxParameterViewController):
    def __init__(
        self,
        constrain_power: BooleanParameter,
        start: IntegerParameter,
        stop: IntegerParameter,
        stride: IntegerParameter,
        num_epochs: IntegerParameter,
    ) -> None:
        super().__init__(
            constrain_power, 'Constrain Power', tool_tip='Whether to constrain probe power'
        )
        self._plan_view_controller = PtyChiOptimizationPlanViewController(
            start, stop, stride, num_epochs
        )

        layout = QFormLayout()
        layout.addRow('Plan:', self._plan_view_controller.get_widget())
        self.get_widget().setLayout(layout)


class PtyChiOrthogonalizeIncoherentModesViewController(CheckableGroupBoxParameterViewController):
    def __init__(
        self,
        orthogonalize_modes: BooleanParameter,
        start: IntegerParameter,
        stop: IntegerParameter,
        stride: IntegerParameter,
        method: StringParameter,
        num_epochs: IntegerParameter,
        enumerators: PtyChiEnumerators,
    ) -> None:
        super().__init__(
            orthogonalize_modes,
            'Orthogonalize Incoherent Modes',
            tool_tip='Whether to orthogonalize incoherent probe modes',
        )
        self._plan_view_controller = PtyChiOptimizationPlanViewController(
            start, stop, stride, num_epochs
        )
        self._method_view_controller = ComboBoxParameterViewController(
            method,
            enumerators.orthogonalization_methods(),
            tool_tip='Method to use for incoherent mode orthogonalization',
        )

        layout = QFormLayout()
        layout.addRow('Plan:', self._plan_view_controller.get_widget())
        layout.addRow('Method:', self._method_view_controller.get_widget())
        self.get_widget().setLayout(layout)


class PtyChiOrthogonalizeOPRModesViewController(CheckableGroupBoxParameterViewController):
    def __init__(
        self,
        orthogonalize_modes: BooleanParameter,
        start: IntegerParameter,
        stop: IntegerParameter,
        stride: IntegerParameter,
        num_epochs: IntegerParameter,
    ) -> None:
        super().__init__(
            orthogonalize_modes,
            'Orthogonalize OPR Modes',
            tool_tip='Whether to orthogonalize OPR modes',
        )
        self._plan_view_controller = PtyChiOptimizationPlanViewController(
            start, stop, stride, num_epochs
        )

        layout = QFormLayout()
        layout.addRow('Plan:', self._plan_view_controller.get_widget())
        self.get_widget().setLayout(layout)


class PtyChiConstrainSupportViewController(CheckableGroupBoxParameterViewController):
    def __init__(
        self,
        constrain_support: BooleanParameter,
        start: IntegerParameter,
        stop: IntegerParameter,
        stride: IntegerParameter,
        threshold: RealParameter,
        num_epochs: IntegerParameter,
    ) -> None:
        super().__init__(
            constrain_support,
            'Constrain Support',
            tool_tip='When enabled, the probe will be shrinkwrapped so that small values are set to zero',
        )
        self._plan_view_controller = PtyChiOptimizationPlanViewController(
            start, stop, stride, num_epochs
        )
        self._threshold_view_controller = DecimalLineEditParameterViewController(
            threshold, tool_tip='Threshold for the probe support constraint'
        )

        layout = QFormLayout()
        layout.addRow('Plan:', self._plan_view_controller.get_widget())
        layout.addRow('Threshold:', self._threshold_view_controller.get_widget())
        self.get_widget().setLayout(layout)


class PtyChiConstrainCenterViewController(CheckableGroupBoxParameterViewController):
    def __init__(
        self,
        constrain_center: BooleanParameter,
        start: IntegerParameter,
        stop: IntegerParameter,
        stride: IntegerParameter,
        num_epochs: IntegerParameter,
        use_intensity_for_mass_centroid: BooleanParameter,
    ) -> None:
        super().__init__(
            constrain_center,
            'Constrain Center',
            tool_tip='When enabled, the probe center of mass will be constrained to the center of the probe array',
        )
        self._plan_view_controller = PtyChiOptimizationPlanViewController(
            start, stop, stride, num_epochs
        )
        self._use_intensity_for_mass_centroid_view_controller = CheckableGroupBoxParameterViewController(
            use_intensity_for_mass_centroid,
            'Use Intensity for Mass Centroid',
            tool_tip='When enabled, the mass centroid will be calculated using the intensity of the probe',
        )

        layout = QFormLayout()
        layout.addRow('Plan:', self._plan_view_controller.get_widget())
        layout.addRow(self._use_intensity_for_mass_centroid_view_controller.get_widget())
        self.get_widget().setLayout(layout)


class PtyChiProbeViewController(CheckableGroupBoxParameterViewController):
    def __init__(
        self,
        settings: PtyChiProbeSettings,
        dm_settings: PtyChiDMSettings | None,
        lsqml_settings: PtyChiLSQMLSettings | None,
        pie_settings: PtyChiPIESettings | None,
        num_epochs: IntegerParameter,
        enumerators: PtyChiEnumerators,
    ) -> None:
        super().__init__(
            settings.is_optimizable, 'Optimize Probe', tool_tip='Whether the probe is optimizable'
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
        self._constrain_probe_power_view_controller = PtyChiConstrainProbePowerViewController(
            settings.constrain_probe_power,
            settings.constrain_probe_power_start,
            settings.constrain_probe_power_stop,
            settings.constrain_probe_power_stride,
            num_epochs,
        )
        self._orthogonalize_incoherent_modes_view_controller = (
            PtyChiOrthogonalizeIncoherentModesViewController(
                settings.orthogonalize_incoherent_modes,
                settings.orthogonalize_incoherent_modes_start,
                settings.orthogonalize_incoherent_modes_stop,
                settings.orthogonalize_incoherent_modes_stride,
                settings.orthogonalize_incoherent_modes_method,
                num_epochs,
                enumerators,
            )
        )
        self._orthogonalize_opr_modes_view_controller = PtyChiOrthogonalizeOPRModesViewController(
            settings.orthogonalize_opr_modes,
            settings.orthogonalize_opr_modes_start,
            settings.orthogonalize_opr_modes_stop,
            settings.orthogonalize_opr_modes_stride,
            num_epochs,
        )
        self._constrain_support_view_controller = PtyChiConstrainSupportViewController(
            settings.constrain_support,
            settings.constrain_support_start,
            settings.constrain_support_stop,
            settings.constrain_support_stride,
            settings.constrain_support_threshold,
            num_epochs,
        )
        self._constrain_center_view_controller = PtyChiConstrainCenterViewController(
            settings.constrain_center,
            settings.constrain_center_start,
            settings.constrain_center_stop,
            settings.constrain_center_stride,
            num_epochs,
            settings.use_intensity_for_mass_centroid,
        )
        self._relax_eigenmode_update_view_controller = DecimalSliderParameterViewController(
            settings.relax_eigenmode_update,
            tool_tip='Relaxation factor for the eigenmode update',
        )

        layout = QFormLayout()
        layout.addRow('Plan:', self._optimization_plan_view_controller.get_widget())
        layout.addRow('Optimizer:', self._optimizer_view_controller.get_widget())
        layout.addRow('Step Size:', self._step_size_view_controller.get_widget())
        layout.addRow(self._constrain_probe_power_view_controller.get_widget())
        layout.addRow(self._orthogonalize_incoherent_modes_view_controller.get_widget())
        layout.addRow(self._orthogonalize_opr_modes_view_controller.get_widget())
        layout.addRow(self._constrain_support_view_controller.get_widget())
        layout.addRow(self._constrain_center_view_controller.get_widget())
        layout.addRow(
            'Relax Eigenmode Update:', self._relax_eigenmode_update_view_controller.get_widget()
        )

        if dm_settings is not None:
            self._inertia_view_controller = DecimalLineEditParameterViewController(
                dm_settings.probe_inertia, tool_tip='Inertia for the probe update'
            )
            layout.addRow('Inertia:', self._inertia_view_controller.get_widget())

        if lsqml_settings is not None:
            self._optimal_step_size_scaler_view_controller = DecimalLineEditParameterViewController(
                lsqml_settings.probe_optimal_step_size_scaler,
                tool_tip='Optimal step size scaler for the probe update',
            )
            layout.addRow(
                'Optimal Step Size Scaler:',
                self._optimal_step_size_scaler_view_controller.get_widget(),
            )

        if pie_settings is not None:
            self._alpha = DecimalSliderParameterViewController(
                pie_settings.probe_alpha, tool_tip='Relaxation factor for the probe update'
            )
            layout.addRow('Alpha:', self._alpha.get_widget())

        self.get_widget().setLayout(layout)

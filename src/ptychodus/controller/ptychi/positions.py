from PyQt5.QtWidgets import QFormLayout, QFrame, QWidget

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.parametric import (
    BooleanParameter,
    IntegerParameter,
    RealParameter,
    StringParameter,
)

from ...model.ptychi import PtyChiEnumerators, PtyChiProbePositionSettings
from ..parametric import (
    CheckBoxParameterViewController,
    CheckableGroupBoxParameterViewController,
    ComboBoxParameterViewController,
    DecimalLineEditParameterViewController,
    DecimalSliderParameterViewController,
    ParameterViewController,
    SpinBoxParameterViewController,
)
from .optimizer import PtyChiOptimizationPlanViewController, PtyChiOptimizerParameterViewController

__all__ = ['PtyChiProbePositionsViewController']


class PtyChiCrossCorrelationViewController(ParameterViewController, Observer):
    def __init__(
        self,
        algorithm: StringParameter,
        scale: IntegerParameter,
        real_space_width: RealParameter,
        probe_threshold: RealParameter,
    ) -> None:
        super().__init__()
        self._algorithm = algorithm
        self._scale_view_controller = SpinBoxParameterViewController(
            scale, tool_tip='Upsampling factor of the cross-correlation in real space.'
        )
        self._real_space_width_view_controller = DecimalLineEditParameterViewController(
            real_space_width, tool_tip='Width of the cross-correlation in real-space'
        )
        self._probe_threshold_view_controller = DecimalSliderParameterViewController(
            probe_threshold, tool_tip='Probe intensity threshold used to calculate the probe mask.'
        )
        self._widget = QFrame()
        self._widget.setFrameShape(QFrame.StyledPanel)

        layout = QFormLayout()
        layout.addRow('Scale:', self._scale_view_controller.get_widget())
        layout.addRow('Real Space Width:', self._real_space_width_view_controller.get_widget())
        layout.addRow('Probe Threshold:', self._probe_threshold_view_controller.get_widget())
        self._widget.setLayout(layout)

        algorithm.add_observer(self)
        self._sync_model_to_view()

    def get_widget(self) -> QWidget:
        return self._widget

    def _sync_model_to_view(self) -> None:
        self._widget.setVisible(self._algorithm.get_value().upper() == 'CROSS_CORRELATION')

    def _update(self, observable: Observable) -> None:
        if observable is self._algorithm:
            self._sync_model_to_view()


class PtyChiUpdateMagnitudeLimitViewController(CheckableGroupBoxParameterViewController):
    def __init__(
        self,
        limit_magnitude_update: BooleanParameter,
        start: IntegerParameter,
        stop: IntegerParameter,
        stride: IntegerParameter,
        magnitude_update_limit: RealParameter,
        num_epochs: IntegerParameter,
    ) -> None:
        super().__init__(
            limit_magnitude_update,
            'Limit Update Magnitude',
            tool_tip='Limit the magnitude of the probe update.',
        )
        self._plan_view_controller = PtyChiOptimizationPlanViewController(
            start, stop, stride, num_epochs
        )
        self._view_controller = DecimalLineEditParameterViewController(
            magnitude_update_limit,
            tool_tip='Magnitude limit of the probe update.',
        )

        layout = QFormLayout()
        layout.addRow('Plan:', self._plan_view_controller.get_widget())
        layout.addRow('Limit:', self._view_controller.get_widget())
        self.get_widget().setLayout(layout)


class PtyChiProbePositionsViewController(CheckableGroupBoxParameterViewController):
    def __init__(
        self,
        settings: PtyChiProbePositionSettings,
        num_epochs: IntegerParameter,
        enumerators: PtyChiEnumerators,
    ) -> None:
        super().__init__(
            settings.is_optimizable,
            'Optimize Probe Positions',
            tool_tip='Whether the probe positions are optimizable.',
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
        self._algorithm_view_controller = ComboBoxParameterViewController(
            settings.position_correction_type,
            enumerators.position_correction_types(),
            tool_tip='Algorithm used to calculate the position correction update.',
        )
        self._cross_correlation_view_controller = PtyChiCrossCorrelationViewController(
            settings.position_correction_type,
            settings.cross_correlation_scale,
            settings.cross_correlation_real_space_width,
            settings.cross_correlation_probe_threshold,
        )
        self._magnitude_update_limit_view_controller = PtyChiUpdateMagnitudeLimitViewController(
            settings.limit_magnitude_update,
            settings.limit_magnitude_update_start,
            settings.limit_magnitude_update_stop,
            settings.limit_magnitude_update_stride,
            settings.magnitude_update_limit,
            num_epochs,
        )
        self._constrain_centroid_view_controller = CheckBoxParameterViewController(
            settings.constrain_centroid,
            'Constrain Centroid',
            tool_tip='Whether to subtract the mean from positions after updating positions.',
        )

        layout = QFormLayout()
        layout.addRow('Plan:', self._optimization_plan_view_controller.get_widget())
        layout.addRow('Optimizer:', self._optimizer_view_controller.get_widget())
        layout.addRow('Step Size:', self._step_size_view_controller.get_widget())
        layout.addRow('Algorithm:', self._algorithm_view_controller.get_widget())
        layout.addRow(self._cross_correlation_view_controller.get_widget())
        layout.addRow(self._magnitude_update_limit_view_controller.get_widget())
        layout.addRow(self._constrain_centroid_view_controller.get_widget())
        self.get_widget().setLayout(layout)

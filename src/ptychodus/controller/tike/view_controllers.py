from ptychodus.api.observer import Observable, Observer
from ptychodus.api.parametric import BooleanParameter, RealParameter

from PyQt5.QtCore import QRegularExpression
from PyQt5.QtGui import QRegularExpressionValidator
from PyQt5.QtWidgets import QComboBox, QFormLayout, QGroupBox, QWidget

from ...model.tike import (
    TikeMultigridSettings,
    TikeObjectCorrectionSettings,
    TikePositionCorrectionSettings,
    TikeProbeCorrectionSettings,
    TikeSettings,
)
from ..parametric import (
    CheckBoxParameterViewController,
    CheckableGroupBoxParameterViewController,
    ComboBoxParameterViewController,
    DecimalLineEditParameterViewController,
    DecimalSliderParameterViewController,
    LineEditParameterViewController,
    ParameterViewController,
    SpinBoxParameterViewController,
)

__all__ = [
    'TikeMultigridViewController',
    'TikeObjectCorrectionViewController',
    'TikePositionCorrectionViewController',
    'TikeProbeCorrectionViewController',
]


class TikeParametersViewController(ParameterViewController, Observer):
    def __init__(self, settings: TikeSettings, *, show_alpha: bool) -> None:
        super().__init__()
        self._settings = settings
        self._num_gpus_view_controller = LineEditParameterViewController(
            settings.num_gpus,
            QRegularExpressionValidator(QRegularExpression('[\\d,]+')),
            tool_tip='The number of GPUs to use. If the number of GPUs is less than the requested number, only workers for the available GPUs are allocated.',
        )
        self._noise_model_view_controller = ComboBoxParameterViewController(
            settings.noise_model,
            settings.get_noise_models(),
            tool_tip='The noise model to use for the cost function.',
        )
        self._num_batch_view_controller = SpinBoxParameterViewController(
            settings.num_batch,
            tool_tip='The dataset is divided into this number of groups where each group is processed sequentially.',
        )
        self._batch_method_view_controller = ComboBoxParameterViewController(
            settings.batch_method,
            settings.get_batch_methods(),
            tool_tip='The name of the batch selection method.',
        )
        self._num_iter_view_controller = SpinBoxParameterViewController(
            settings.num_iter, tool_tip='The number of epochs to process before returning.'
        )
        self._convergence_window_view_controller = SpinBoxParameterViewController(
            settings.convergence_window,
            tool_tip='The number of epochs to consider for convergence monitoring. Set to any value less than 2 to disable.',
        )
        self._alpha_view_controller = DecimalSliderParameterViewController(
            settings.alpha, tool_tip='RPIE becomes EPIE when this parameter is 1.'
        )
        self._log_level_combo_box = QComboBox()

        for model in settings.get_log_levels():
            self._log_level_combo_box.addItem(model)

        self._log_level_combo_box.textActivated.connect(settings.set_log_level)

        self._widget = QGroupBox('Tike Parameters')

        layout = QFormLayout()
        layout.addRow('Number of GPUs:', self._num_gpus_view_controller.get_widget())
        layout.addRow('Noise Model:', self._noise_model_view_controller.get_widget())
        layout.addRow('Number of Batches:', self._num_batch_view_controller.get_widget())
        layout.addRow('Batch Method:', self._batch_method_view_controller.get_widget())
        layout.addRow('Number of Iterations:', self._num_iter_view_controller.get_widget())
        layout.addRow('Convergence Window:', self._convergence_window_view_controller.get_widget())

        if show_alpha:
            layout.addRow('Alpha:', self._alpha_view_controller.get_widget())

        layout.addRow('Log Level:', self._log_level_combo_box)
        self._widget.setLayout(layout)

        self._sync_model_to_view()
        self._settings.add_observer(self)

    def get_widget(self) -> QWidget:
        return self._widget

    def _sync_model_to_view(self) -> None:
        self._log_level_combo_box.setCurrentText(self._settings.get_log_level())

    def _update(self, observable: Observable) -> None:
        if observable is self._settings:
            self._sync_model_to_view()


class TikeMultigridViewController(CheckableGroupBoxParameterViewController):
    def __init__(self, settings: TikeMultigridSettings) -> None:
        super().__init__(settings.use_multigrid, 'Multigrid')
        self._use_multigrid = settings.use_multigrid
        self._num_levels_controller = SpinBoxParameterViewController(
            settings.num_levels,
            tool_tip='The number of times to reduce the problem by a factor of two.',
        )

        layout = QFormLayout()
        layout.addRow('Number of Levels:', self._num_levels_controller.get_widget())
        self.get_widget().setLayout(layout)


class TikeAdaptiveMomentViewController(CheckableGroupBoxParameterViewController):
    def __init__(
        self, use_adaptive_moment: BooleanParameter, mdecay: RealParameter, vdecay: RealParameter
    ) -> None:
        super().__init__(use_adaptive_moment, 'Adaptive Moment')
        self._use_adaptive_moment = use_adaptive_moment
        self._mdecay_view_controller = DecimalSliderParameterViewController(
            mdecay, tool_tip='The proportion of the first moment that is previous first moments.'
        )
        self._vdecay_view_controller = DecimalSliderParameterViewController(
            vdecay, tool_tip='The proportion of the second moment that is previous second moments.'
        )

        layout = QFormLayout()
        layout.addRow('M Decay:', self._mdecay_view_controller.get_widget())
        layout.addRow('V Decay:', self._vdecay_view_controller.get_widget())
        self.get_widget().setLayout(layout)


class TikeObjectCorrectionViewController(CheckableGroupBoxParameterViewController):
    def __init__(self, settings: TikeObjectCorrectionSettings) -> None:
        super().__init__(settings.use_object_correction, 'Object Correction')
        self._positivity_constraint_view_controller = DecimalSliderParameterViewController(
            settings.positivity_constraint
        )
        self._smoothness_constraint_view_controller = DecimalSliderParameterViewController(
            settings.smoothness_constraint
        )
        self._adaptive_moment_view_controller = TikeAdaptiveMomentViewController(
            settings.use_adaptive_moment, settings.mdecay, settings.vdecay
        )
        self._use_magnitude_clipping_view_controller = CheckBoxParameterViewController(
            settings.use_magnitude_clipping,
            'Magnitude Clipping',
            tool_tip='Forces the object magnitude to be <= 1.',
        )

        layout = QFormLayout()
        layout.addRow(
            'Positivity Constraint:', self._positivity_constraint_view_controller.get_widget()
        )
        layout.addRow(
            'Smoothness Constraint:', self._smoothness_constraint_view_controller.get_widget()
        )
        layout.addRow(self._adaptive_moment_view_controller.get_widget())
        layout.addRow(self._use_magnitude_clipping_view_controller.get_widget())
        self.get_widget().setLayout(layout)


class TikeProbeSupportViewController(CheckableGroupBoxParameterViewController):
    def __init__(self, settings: TikeProbeCorrectionSettings) -> None:
        super().__init__(settings.use_finite_probe_support, 'Finite Probe Support')
        self._weight_view_controller = DecimalLineEditParameterViewController(
            settings.probe_support_weight, tool_tip='Weight of the finite probe constraint.'
        )
        self._radius_view_controller = DecimalSliderParameterViewController(
            settings.probe_support_radius,
            tool_tip='Radius of probe support as fraction of probe grid.',
        )
        self._degree_view_controller = DecimalLineEditParameterViewController(
            settings.probe_support_degree,
            tool_tip='Degree of the supergaussian defining the probe support.',
        )

        layout = QFormLayout()
        layout.addRow('Weight:', self._weight_view_controller.get_widget())
        layout.addRow('Radius:', self._radius_view_controller.get_widget())
        layout.addRow('Degree:', self._degree_view_controller.get_widget())
        self.get_widget().setLayout(layout)


class TikeProbeCorrectionViewController(CheckableGroupBoxParameterViewController):
    def __init__(self, settings: TikeProbeCorrectionSettings) -> None:
        super().__init__(settings.use_probe_correction, 'Probe Correction')
        self._force_sparsity_view_controller = DecimalSliderParameterViewController(
            settings.force_sparsity, tool_tip='Forces this proportion of zero elements.'
        )
        self._force_orthogonality_view_controller = CheckBoxParameterViewController(
            settings.force_orthogonality,
            'Force Orthogonality',
            tool_tip='Forces probes to be orthogonal each iteration.',
        )
        self._force_centered_intensity_view_controller = CheckBoxParameterViewController(
            settings.force_centered_intensity,
            'Force Centered Intensity',
            tool_tip='Forces the probe intensity to be centered.',
        )
        self._support_view_controller = TikeProbeSupportViewController(settings)
        self._adaptive_moment_view_controller = TikeAdaptiveMomentViewController(
            settings.use_adaptive_moment, settings.mdecay, settings.vdecay
        )
        self._additional_probe_penalty_view_controller = DecimalLineEditParameterViewController(
            settings.additional_probe_penalty,
            tool_tip='Penalty applied to the last probe for existing.',
        )

        layout = QFormLayout()
        layout.addRow('Force Sparsity:', self._force_sparsity_view_controller.get_widget())
        layout.addRow(self._force_orthogonality_view_controller.get_widget())
        layout.addRow(self._force_centered_intensity_view_controller.get_widget())
        layout.addRow(self._support_view_controller.get_widget())
        layout.addRow(self._adaptive_moment_view_controller.get_widget())
        layout.addRow(
            'Additional Probe Penalty:', self._additional_probe_penalty_view_controller.get_widget()
        )
        self.get_widget().setLayout(layout)


class TikePositionCorrectionViewController(CheckableGroupBoxParameterViewController):
    def __init__(self, settings: TikePositionCorrectionSettings) -> None:
        super().__init__(settings.use_position_correction, 'Position Correction')
        self._use_position_regularization_view_controller = CheckBoxParameterViewController(
            settings.use_position_regularization,
            'Use Regularization',
            tool_tip='Whether the positions are constrained to fit a random error plus affine error model.',
        )
        self._adaptive_moment_view_controller = TikeAdaptiveMomentViewController(
            settings.use_adaptive_moment, settings.mdecay, settings.vdecay
        )
        self._update_magnitude_limit_view_controller = DecimalLineEditParameterViewController(
            settings.update_magnitude_limit,
            tool_tip='When set to a positive number, x and y update magnitudes are clipped (limited) to this value.',
        )

        layout = QFormLayout()
        layout.addRow(self._use_position_regularization_view_controller.get_widget())
        layout.addRow(self._adaptive_moment_view_controller.get_widget())
        layout.addRow(
            'Update Magnitude Limit:', self._update_magnitude_limit_view_controller.get_widget()
        )
        self.get_widget().setLayout(layout)

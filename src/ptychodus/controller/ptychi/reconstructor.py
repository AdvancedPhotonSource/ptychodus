from PyQt5.QtWidgets import (
    QButtonGroup,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.parametric import BooleanParameter, RealParameter

from ...model.ptychi import (
    PtyChiAutodiffSettings,
    PtyChiBHSettings,
    PtyChiDMSettings,
    PtyChiDeviceRepository,
    PtyChiEnumerators,
    PtyChiLSQMLSettings,
    PtyChiSettings,
)
from ..parametric import (
    CheckBoxParameterViewController,
    CheckableGroupBoxParameterViewController,
    ComboBoxParameterViewController,
    DecimalLineEditParameterViewController,
    DecimalSliderParameterViewController,
    ParameterViewController,
    SpinBoxParameterViewController,
)

__all__ = ['PtyChiReconstructorViewController']


class PtyChiDeviceViewController(CheckableGroupBoxParameterViewController):
    def __init__(
        self,
        use_devices: BooleanParameter,
        repository: PtyChiDeviceRepository,
        *,
        tool_tip: str = '',
    ) -> None:
        super().__init__(use_devices, 'Use Devices', tool_tip=tool_tip)
        layout = QVBoxLayout()

        for device in repository:
            device_label = QLabel(device)
            layout.addWidget(device_label)

        self.get_widget().setLayout(layout)


class PtyChiPrecisionParameterViewController(ParameterViewController, Observer):
    def __init__(self, use_double_precision: BooleanParameter, *, tool_tip: str = '') -> None:
        super().__init__()
        self._use_double_precision = use_double_precision
        self._single_precision_button = QRadioButton('Single')
        self._double_precision_button = QRadioButton('Double')
        self._button_group = QButtonGroup()
        self._widget = QWidget()

        self._single_precision_button.setToolTip('Compute using single precision')
        self._double_precision_button.setToolTip('Compute using double precision')

        if tool_tip:
            self._widget.setToolTip(tool_tip)

        self._button_group.addButton(self._single_precision_button, 1)
        self._button_group.addButton(self._double_precision_button, 2)
        self._button_group.setExclusive(True)
        self._button_group.idToggled.connect(self._sync_view_to_model)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._single_precision_button)
        layout.addWidget(self._double_precision_button)
        layout.addStretch()
        self._widget.setLayout(layout)

        self._sync_model_to_view()
        use_double_precision.add_observer(self)

    def get_widget(self) -> QWidget:
        return self._widget

    def _sync_view_to_model(self, tool_id: int, checked: bool) -> None:
        if tool_id == 2:
            self._use_double_precision.set_value(checked)

    def _sync_model_to_view(self) -> None:
        button = self._button_group.button(2 if self._use_double_precision.get_value() else 1)

        if button is not None:
            button.setChecked(True)

    def _update(self, observable: Observable) -> None:
        if observable is self._use_double_precision:
            self._sync_model_to_view()


class PtyChiMomentumAccelerationGradientMixingFactorViewController(
    CheckableGroupBoxParameterViewController
):
    def __init__(
        self,
        use_gradient_mixing_factor: BooleanParameter,
        gradient_mixing_factor: RealParameter,
        *,
        tool_tip: str = '',
    ) -> None:
        super().__init__(
            use_gradient_mixing_factor,
            'Use Gradient Mixing Factor',
            tool_tip='Controls how the current gradient is mixed with the accumulated velocity in LSQML momentum acceleration',
        )
        self._gradient_mixing_factor_view_controller = DecimalLineEditParameterViewController(
            gradient_mixing_factor, tool_tip=tool_tip
        )

        layout = QVBoxLayout()
        layout.addWidget(self._gradient_mixing_factor_view_controller.get_widget())
        self.get_widget().setLayout(layout)


class PtyChiReconstructorViewController(ParameterViewController):
    def __init__(
        self,
        settings: PtyChiSettings,
        autodiff_settings: PtyChiAutodiffSettings | None,
        bh_settings: PtyChiBHSettings | None,
        dm_settings: PtyChiDMSettings | None,
        lsqml_settings: PtyChiLSQMLSettings | None,
        enumerators: PtyChiEnumerators,
        repository: PtyChiDeviceRepository,
    ) -> None:
        super().__init__()
        self._num_epochs_view_controller = SpinBoxParameterViewController(
            settings.num_epochs, tool_tip='Number of epochs to run'
        )
        self._num_sync_epochs_view_controller = SpinBoxParameterViewController(
            settings.num_sync_epochs, tool_tip='Number of epochs between updates'
        )
        self._batch_size_view_controller = SpinBoxParameterViewController(
            settings.batch_size, tool_tip='Number of data to process in each minibatch'
        )
        self._batching_mode_view_controller = ComboBoxParameterViewController(
            settings.batching_mode, enumerators.batching_modes(), tool_tip='Batching mode to use'
        )
        self._compact_mode_update_clustering_view_controller = SpinBoxParameterViewController(
            settings.compact_mode_update_clustering,
            tool_tip='When greater than zero, the number of epochs between updating clusters in compact batching mode',
        )
        self._device_view_controller = PtyChiDeviceViewController(
            settings.use_devices, repository, tool_tip='Default device to use for computation'
        )
        self._compute_precision_view_controller = PtyChiPrecisionParameterViewController(
            settings.use_double_precision,
            tool_tip='Floating point precision to use for computation',
        )
        self._fft_precision_view_controller = PtyChiPrecisionParameterViewController(
            settings.use_double_precision_for_fft,
            tool_tip='Floating point precision to use for critical FFT operations',
        )
        self.allow_nondeterministic_algorithms_view_controller = CheckBoxParameterViewController(
            settings.allow_nondeterministic_algorithms,
            'Allow Nondeterministic Algorithms',
            tool_tip='When checked, nondeterministic algorithms will be used. This may lead to different results on different runs',
        )

        self._use_low_memory_view_controller = CheckBoxParameterViewController(
            settings.use_low_memory_mode,
            'Use Low Memory Mode',
            tool_tip='When checked, forward propagation of ptychography will be done using less vectorized code. This reduces the speed, but also lowers memory usage',
        )
        self._pad_for_shift_view_controller = SpinBoxParameterViewController(
            settings.pad_for_shift,
            tool_tip='Number of pixels to pad arrays (with border values) before shifting',
        )

        self._use_far_field_propagation_view_controller = CheckBoxParameterViewController(
            settings.use_far_field_propagation,
            'Use Far Field Propagation',
            tool_tip='When checked, far field propagation will be used instead of near field propagation',
        )
        self._fft_shift_diffraction_patterns_view_controller = CheckBoxParameterViewController(
            settings.fft_shift_diffraction_patterns,
            'FFT Shift Diffraction Patterns',
            tool_tip='When checked, the diffraction patterns will be FFT-shifted',
        )
        self._save_data_on_device_view_controller = CheckBoxParameterViewController(
            settings.save_data_on_device,
            'Save Data on Device',
            tool_tip='When checked, diffraction data will be saved on the device',
        )
        self._diffraction_pattern_blur_view_controller = CheckableGroupBoxParameterViewController(
            settings.enable_diffraction_pattern_blur,
            'Blur Diffraction Patterns',
            tool_tip='When enabled, diffraction patterns will be blurred with a Gaussian kernel',
        )
        self._diffraction_pattern_blur_sigma_view_controller = (
            DecimalLineEditParameterViewController(
                settings.diffraction_pattern_blur_sigma,
                tool_tip='Standard deviation of the Gaussian blur kernel in pixels',
            )
        )
        self._exclude_measured_pixels_below_view_controller = (
            CheckableGroupBoxParameterViewController(
                settings.enable_exclude_measured_pixels_below,
                'Exclude Measured Pixels Below',
                tool_tip='When enabled, measured pixels below the threshold will be excluded',
            )
        )
        self._exclude_measured_pixels_below_value_view_controller = (
            DecimalLineEditParameterViewController(
                settings.exclude_measured_pixels_below,
                tool_tip='Intensity threshold below which measured pixels are excluded',
            )
        )
        _diffraction_blur_layout = QVBoxLayout()
        _diffraction_blur_layout.addWidget(
            self._diffraction_pattern_blur_sigma_view_controller.get_widget()
        )
        self._diffraction_pattern_blur_view_controller.get_widget().setLayout(
            _diffraction_blur_layout
        )
        _exclude_pixels_layout = QVBoxLayout()
        _exclude_pixels_layout.addWidget(
            self._exclude_measured_pixels_below_value_view_controller.get_widget()
        )
        self._exclude_measured_pixels_below_view_controller.get_widget().setLayout(
            _exclude_pixels_layout
        )
        self._widget = QGroupBox('Reconstructor')

        layout = QFormLayout()
        layout.addRow('Number of Epochs:', self._num_epochs_view_controller.get_widget())
        layout.addRow('Update Interval:', self._num_sync_epochs_view_controller.get_widget())

        if dm_settings is None:
            layout.addRow('Batch Size:', self._batch_size_view_controller.get_widget())
            layout.addRow('Batch Mode:', self._batching_mode_view_controller.get_widget())
            layout.addRow(
                'Update Clustering:',
                self._compact_mode_update_clustering_view_controller.get_widget(),
            )

        layout.addRow(self._device_view_controller.get_widget())
        layout.addRow('Compute Precision:', self._compute_precision_view_controller.get_widget())
        layout.addRow('FFT Precision:', self._fft_precision_view_controller.get_widget())
        layout.addRow(self.allow_nondeterministic_algorithms_view_controller.get_widget())

        layout.addRow(self._use_low_memory_view_controller.get_widget())
        layout.addRow('Pad For Shift:', self._pad_for_shift_view_controller.get_widget())

        layout.addRow(self._use_far_field_propagation_view_controller.get_widget())
        layout.addRow(self._fft_shift_diffraction_patterns_view_controller.get_widget())
        layout.addRow(self._save_data_on_device_view_controller.get_widget())
        layout.addRow(self._diffraction_pattern_blur_view_controller.get_widget())
        layout.addRow(self._exclude_measured_pixels_below_view_controller.get_widget())

        if autodiff_settings is not None:
            self._loss_function_view_controller = ComboBoxParameterViewController(
                autodiff_settings.loss_function,
                enumerators.loss_functions(),
                tool_tip='Loss function to optimize',
            )
            layout.addRow('Loss Function:', self._loss_function_view_controller.get_widget())

            self._forward_model_class_view_controller = ComboBoxParameterViewController(
                autodiff_settings.forward_model_class,
                enumerators.forward_models(),
                tool_tip='Forward model class',
            )
            layout.addRow('Forward Model:', self._forward_model_class_view_controller.get_widget())

        if dm_settings is not None:
            self._exit_wave_update_relaxation_view_controller = (
                DecimalSliderParameterViewController(
                    dm_settings.exit_wave_update_relaxation,
                    tool_tip='Relaxation multiplier for the exit wave update',
                )
            )
            layout.addRow(
                'Exit Wave Update Relaxation:',
                self._exit_wave_update_relaxation_view_controller.get_widget(),
            )

            self._chunk_length_view_controller = SpinBoxParameterViewController(
                dm_settings.chunk_length,
                tool_tip='Number of probe positions used in each chunk of the exit wave update loop',
            )
            layout.addRow('Chunk Length:', self._chunk_length_view_controller.get_widget())

        if lsqml_settings is not None:
            self._noise_model_view_controller = ComboBoxParameterViewController(
                lsqml_settings.noise_model,
                enumerators.noise_models(),
                tool_tip='Noise model to use',
            )
            layout.addRow('Noise Model:', self._noise_model_view_controller.get_widget())

            self._gaussian_noise_deviation_view_controller = DecimalLineEditParameterViewController(
                lsqml_settings.gaussian_noise_deviation,
                tool_tip='Standard deviation of the Gaussian noise',
            )
            layout.addRow(
                'Gaussian Noise Deviation:',
                self._gaussian_noise_deviation_view_controller.get_widget(),
            )

            self._single_slice_solve_object_probe_step_size_jointly_view_controller = CheckBoxParameterViewController(
                lsqml_settings.single_slice_solve_object_probe_step_size_jointly,
                'Solve Object/Probe Step Size Jointly (Single Slice)',
                tool_tip='When checked, object and probe step lengths are solved simultaneously in single-slice mode',
            )
            layout.addRow(
                self._single_slice_solve_object_probe_step_size_jointly_view_controller.get_widget()
            )

            self._multislice_solve_object_probe_step_size_jointly_view_controller = CheckBoxParameterViewController(
                lsqml_settings.multislice_solve_object_probe_step_size_jointly,
                'Solve Object/Probe Step Size Jointly (Multislice)',
                tool_tip='When checked, object and probe step lengths are solved simultaneously in multislice mode',
            )
            layout.addRow(
                self._multislice_solve_object_probe_step_size_jointly_view_controller.get_widget()
            )

            self._solve_step_sizes_only_using_first_probe_mode_view_controller = CheckBoxParameterViewController(
                lsqml_settings.solve_step_sizes_only_using_first_probe_mode,
                'Solve Step Sizes Only Using First Probe Mode',
                tool_tip='When checked, the step sizes will be calculated using only the first probe mode',
            )
            layout.addRow(
                self._solve_step_sizes_only_using_first_probe_mode_view_controller.get_widget()
            )

            self._momentum_acceleration_gain_view_controller = (
                DecimalLineEditParameterViewController(
                    lsqml_settings.momentum_acceleration_gain,
                    tool_tip='Gain of momentum accleration',
                )
            )
            layout.addRow(
                'Momentum Acceleration Gain:',
                self._momentum_acceleration_gain_view_controller.get_widget(),
            )

            self._momentum_acceleration_gradient_mixing_factor_view_controller = PtyChiMomentumAccelerationGradientMixingFactorViewController(
                lsqml_settings.auto_momentum_acceleration_gradient_mixing_factor,
                lsqml_settings.momentum_acceleration_gradient_mixing_factor,
                tool_tip='Controls how the current gradient is mixed with the accumulated velocity in LSQML momentum acceleration',
            )
            layout.addRow(
                self._momentum_acceleration_gradient_mixing_factor_view_controller.get_widget()
            )

            self._rescale_probe_intensity_in_first_epoch_view_controller = CheckBoxParameterViewController(
                lsqml_settings.rescale_probe_intensity_in_first_epoch,
                'Rescale Probe Intensity In First Epoch',
                tool_tip='When checked, the probe intensity will be rescaled on the first epoch',
            )
            layout.addRow(self._rescale_probe_intensity_in_first_epoch_view_controller.get_widget())

            self._preconditioning_damping_factor_view_controller = (
                DecimalLineEditParameterViewController(
                    lsqml_settings.preconditioning_damping_factor,
                    tool_tip='Damping factor for preconditioning the object update',
                )
            )
            layout.addRow(
                'Preconditioning Damping Factor:',
                self._preconditioning_damping_factor_view_controller.get_widget(),
            )

            self._probe_position_momentum_acceleration_gain_view_controller = (
                DecimalLineEditParameterViewController(
                    lsqml_settings.probe_position_momentum_acceleration_gain,
                    tool_tip='Gain of probe position momentum acceleration',
                )
            )
            layout.addRow(
                'Position Momentum Gain:',
                self._probe_position_momentum_acceleration_gain_view_controller.get_widget(),
            )

            self._probe_position_momentum_gradient_mixing_factor_view_controller = PtyChiMomentumAccelerationGradientMixingFactorViewController(
                lsqml_settings.probe_position_auto_momentum_gradient_mixing_factor,
                lsqml_settings.probe_position_momentum_acceleration_gradient_mixing_factor,
                tool_tip='Controls how the current gradient is mixed with the accumulated velocity in probe position momentum acceleration',
            )
            layout.addRow(
                self._probe_position_momentum_gradient_mixing_factor_view_controller.get_widget()
            )

            self._probe_position_momentum_acceleration_memory_view_controller = SpinBoxParameterViewController(
                lsqml_settings.probe_position_momentum_acceleration_memory,
                tool_tip='Number of past gradients to use in probe position momentum acceleration',
            )
            layout.addRow(
                'Position Momentum Memory:',
                self._probe_position_momentum_acceleration_memory_view_controller.get_widget(),
            )

        if bh_settings is not None:
            self._bh_method_view_controller = ComboBoxParameterViewController(
                bh_settings.method,
                iter(['GD', 'CG']),
                tool_tip='BH reconstruction algorithm: GD (Gradient Descent) or CG (Conjugate Gradients)',
            )
            layout.addRow('BH Method:', self._bh_method_view_controller.get_widget())

            self._bh_probe_rho_view_controller = DecimalLineEditParameterViewController(
                bh_settings.probe_rho,
                tool_tip='Rho parameter for the probe update in BH',
            )
            layout.addRow('BH Probe Rho:', self._bh_probe_rho_view_controller.get_widget())

            self._bh_probe_position_rho_view_controller = DecimalLineEditParameterViewController(
                bh_settings.probe_position_rho,
                tool_tip='Rho parameter for probe position update in BH',
            )
            layout.addRow(
                'BH Position Rho:', self._bh_probe_position_rho_view_controller.get_widget()
            )

        self._widget.setLayout(layout)

    def get_widget(self) -> QWidget:
        return self._widget

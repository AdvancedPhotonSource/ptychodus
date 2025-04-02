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
    PtyChiDMSettings,
    PtyChiDeviceRepository,
    PtyChiEnumerators,
    PtyChiLSQMLSettings,
    PtyChiReconstructorSettings,
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

        self._single_precision_button.setToolTip('Compute using single precision.')
        self._double_precision_button.setToolTip('Compute using double precision.')

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
    ) -> None:
        super().__init__(
            use_gradient_mixing_factor,
            'Use Gradient Mixing Factor',
            tool_tip='Controls how the current gradient is mixed with the accumulated velocity in LSQML momentum acceleration.',
        )
        self._gradient_mixing_factor_view_controller = DecimalLineEditParameterViewController(
            gradient_mixing_factor
        )

        layout = QVBoxLayout()
        layout.addWidget(self._gradient_mixing_factor_view_controller.get_widget())
        self.get_widget().setLayout(layout)


class PtyChiReconstructorViewController(ParameterViewController):
    def __init__(
        self,
        settings: PtyChiReconstructorSettings,
        autodiff_settings: PtyChiAutodiffSettings | None,
        dm_settings: PtyChiDMSettings | None,
        lsqml_settings: PtyChiLSQMLSettings | None,
        enumerators: PtyChiEnumerators,
        repository: PtyChiDeviceRepository,
    ) -> None:
        super().__init__()
        self._num_epochs_view_controller = SpinBoxParameterViewController(
            settings.num_epochs, tool_tip='Number of epochs to run.'
        )
        self._batch_size_view_controller = SpinBoxParameterViewController(
            settings.batch_size, tool_tip='Number of data to process in each minibatch.'
        )
        self._batching_mode_view_controller = ComboBoxParameterViewController(
            settings.batching_mode, enumerators.batching_modes(), tool_tip='Batching mode to use.'
        )
        self._compact_mode_update_clustering = SpinBoxParameterViewController(
            settings.compact_mode_update_clustering,
            tool_tip='Number of epochs between updating clusters.',
        )
        self._precision_view_controller = PtyChiPrecisionParameterViewController(
            settings.use_double_precision,
            tool_tip='Floating point precision to use for computation.',
        )
        self._device_view_controller = PtyChiDeviceViewController(
            settings.use_devices, repository, tool_tip='Default device to use for computation.'
        )
        self._use_low_memory_view_controller = CheckBoxParameterViewController(
            settings.use_low_memory_mode,
            'Use Low Memory Mode',
            tool_tip='When checked, forward propagation of ptychography will be done using less vectorized code. This reduces the speed, but also lowers memory usage.',
        )
        self._save_data_on_device_view_controller = CheckBoxParameterViewController(
            settings.save_data_on_device,
            'Save Data on Device',
            tool_tip='When checked, diffraction data will be saved on the device.',
        )
        self._widget = QGroupBox('Reconstructor')

        layout = QFormLayout()
        layout.addRow('Number of Epochs:', self._num_epochs_view_controller.get_widget())
        layout.addRow('Batch Size:', self._batch_size_view_controller.get_widget())
        layout.addRow('Batch Mode:', self._batching_mode_view_controller.get_widget())
        layout.addRow('Update Clustering:', self._compact_mode_update_clustering.get_widget())

        if repository:
            layout.addRow(self._device_view_controller.get_widget())

        layout.addRow('Precision:', self._precision_view_controller.get_widget())
        layout.addRow(self._use_low_memory_view_controller.get_widget())

        if autodiff_settings is not None:
            self._loss_function_view_controller = ComboBoxParameterViewController(
                autodiff_settings.loss_function, enumerators.loss_functions()
            )
            layout.addRow('Loss Function:', self._loss_function_view_controller.get_widget())

            self._forward_model_class_view_controller = ComboBoxParameterViewController(
                autodiff_settings.forward_model_class, enumerators.forward_models()
            )
            layout.addRow('Forward Model:', self._forward_model_class_view_controller.get_widget())

        if dm_settings is not None:
            self._exit_wave_update_relaxation_view_controller = (
                DecimalSliderParameterViewController(dm_settings.exit_wave_update_relaxation)
            )
            layout.addRow(
                'Exit Wave Update Relaxation:',
                self._exit_wave_update_relaxation_view_controller.get_widget(),
            )

            self._chunk_length_view_controller = SpinBoxParameterViewController(
                dm_settings.chunk_length
            )
            layout.addRow('Chunk Length:', self._chunk_length_view_controller.get_widget())

        if lsqml_settings is not None:
            self._noise_model_view_controller = ComboBoxParameterViewController(
                lsqml_settings.noise_model, enumerators.noise_models()
            )
            layout.addRow('Noise Model:', self._noise_model_view_controller.get_widget())

            self._gaussian_noise_deviation_view_controller = DecimalLineEditParameterViewController(
                lsqml_settings.gaussian_noise_deviation
            )
            layout.addRow(
                'Gaussian Noise Deviation:',
                self._gaussian_noise_deviation_view_controller.get_widget(),
            )

            self._solve_object_probe_step_size_jointly_for_first_slice_in_multislice_view_controller = CheckBoxParameterViewController(
                lsqml_settings.solve_object_probe_step_size_jointly_for_first_slice_in_multislice,
                'Solve Object Probe Step Size Jointly For First Slice In Multislice',
            )
            layout.addRow(
                self._solve_object_probe_step_size_jointly_for_first_slice_in_multislice_view_controller.get_widget()
            )

            self._solve_step_sizes_only_using_first_probe_mode_view_controller = (
                CheckBoxParameterViewController(
                    lsqml_settings.solve_step_sizes_only_using_first_probe_mode,
                    'Solve Step Sizes Only Using First Probe Mode',
                )
            )
            layout.addRow(
                self._solve_step_sizes_only_using_first_probe_mode_view_controller.get_widget()
            )

            self._momentum_acceleration_gain_view_controller = (
                DecimalLineEditParameterViewController(lsqml_settings.momentum_acceleration_gain)
            )
            layout.addRow(
                'Momentum Acceleration Gain:',
                self._momentum_acceleration_gain_view_controller.get_widget(),
            )

            self._momentum_acceleration_gradient_mixing_factor_view_controller = (
                PtyChiMomentumAccelerationGradientMixingFactorViewController(
                    lsqml_settings.use_momentum_acceleration_gradient_mixing_factor,
                    lsqml_settings.momentum_acceleration_gradient_mixing_factor,
                )
            )
            layout.addRow(
                self._momentum_acceleration_gradient_mixing_factor_view_controller.get_widget()
            )

        self._widget.setLayout(layout)

    def get_widget(self) -> QWidget:
        return self._widget

from PyQt5.QtGui import QValidator
from PyQt5.QtWidgets import QSpinBox, QWidget

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.parametric import IntegerParameter

from ..model.ptychopinn.core import PtychoPINNReconstructorLibrary
from .data import FileDialogFactory
from .parametric import ParameterViewBuilder, ParameterViewController
from .reconstructor import ReconstructorViewControllerFactory


class PowerTwoSpinBox(QSpinBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

    def stepBy(self, steps: int) -> None:  # noqa: N802
        if steps < 0:
            self.setValue(self.value() // (1 << -steps))
        elif steps > 0:
            self.setValue(self.value() * (1 << steps))

    def validate(self, input: str | None, pos: int) -> tuple[QValidator.State, str, int]:
        try:
            value = int(input)
        except ValueError:
            pass
        else:
            if value > 0:
                is_pow2 = (value & (value - 1)) == 0

                if is_pow2:
                    return QValidator.State.Acceptable, input, pos

        return QValidator.State.Intermediate, input, pos


class PowerTwoSpinBoxParameterViewController(ParameterViewController, Observer):
    def __init__(self, parameter: IntegerParameter, *, tool_tip: str = '') -> None:
        super().__init__()
        self._parameter = parameter
        self._widget = PowerTwoSpinBox()

        if tool_tip:
            self._widget.setToolTip(tool_tip)

        self._sync_model_to_view()
        self._widget.valueChanged.connect(parameter.set_value)
        parameter.add_observer(self)

    def get_widget(self) -> QWidget:
        return self._widget

    def _sync_model_to_view(self) -> None:
        minimum = self._parameter.get_minimum()
        maximum = self._parameter.get_maximum()

        if minimum is None:
            raise ValueError('Minimum not provided!')

        if maximum is None:
            raise ValueError('Maximum not provided!')

        self._widget.blockSignals(True)
        self._widget.setRange(minimum, maximum)
        self._widget.setValue(self._parameter.get_value())
        self._widget.blockSignals(False)

    def _update(self, observable: Observable) -> None:
        if observable is self._parameter:
            self._sync_model_to_view()


class PtychoPINNViewControllerFactory(ReconstructorViewControllerFactory):
    def __init__(
        self, model: PtychoPINNReconstructorLibrary, file_dialog_factory: FileDialogFactory
    ) -> None:
        super().__init__()
        self._model = model
        self._file_dialog_factory = file_dialog_factory

    @property
    def backend_name(self) -> str:
        return 'PtychoPINN'

    def create_view_controller(self, reconstructor_name: str) -> QWidget:
        is_pinn = reconstructor_name.lower() == 'pinn'
        builder = ParameterViewBuilder(self._file_dialog_factory)
        enumerators = self._model.enumerators

        model_group = 'Model'
        model_settings = self._model.model_settings
        builder.add_spin_box(
            model_settings.n_filters_scale,
            'Num. Filters Scale Factor:',
            tool_tip='Scale factor for number of filters',
            group=model_group,
        )

        if is_pinn:
            builder.add_spin_box(
                model_settings.gridsize,
                'Grid Size:',
                tool_tip='Controls number of images processed per solution region (e.g., gridsize=2 means 2^2=4 images at a time)',
                group=model_group,
            )
            builder.add_combo_box(
                model_settings.amp_activation,
                enumerators.get_amp_activations(),
                'Amplitude Activation Function:',
                group=model_group,
            )
            builder.add_check_box(
                model_settings.object_big,
                'Object Big',
                group=model_group,
                tool_tip='Enables a separate real-space reconstruction for each input diffraction image '
                'and an averaging / overlap constraint step. If False, no explicit averaging is performed '
                'and the decoders return a single real space image instead of `gridsize**2` images. '
                'Typically left True.',
            )
            builder.add_check_box(
                model_settings.probe_big,
                'Probe Big',
                group=model_group,
                tool_tip='If True, enables a low-resolution reconstruction of the outer region of the NxN real-space grid. '
                'This technically violates the zero-padding / oversampling condition, '
                'but may be needed if the probe illumination has wide tails. '
                'Has no effect unless pad_object is True.',
            )
            builder.add_check_box(
                model_settings.probe_mask,
                'Probe Mask',
                group=model_group,
                tool_tip='Whether to apply circular mask to the probe function. '
                "If toggling this changes the reconstruction, it's likely that there are edge / real space truncation artifacts. "
                'Should be used with pad_object = False.',
            )
            builder.add_check_box(
                model_settings.pad_object,
                'Pad Object',
                group=model_group,
                tool_tip='Whether to reconstruct the full real space grid (False) or restrict to N/2 x N/2 (True). '
                'True strictly enforces the necessary reciprocal space oversampling, '
                'but may cause truncation issues for probe amplitudes with long tails. '
                'This truncation can be mitigated by setting probe_big, '
                'which uses a small number of CNN filters to generate a low-resolution reconstruction of the outer region. '
                'Typically left True.',
            )
            builder.add_decimal_line_edit(
                model_settings.probe_scale,
                'Probe Scale Factor:',
                group=model_group,
                tool_tip='Scaling factor for the probe amplitude. ',
            )
            builder.add_decimal_line_edit(
                model_settings.gaussian_smoothing_sigma,
                'Gaussian Smoothing Sigma:',
                group=model_group,
                tool_tip='Standard deviation for Gaussian smoothing of probe illumination.  '
                'Increase from 0 to reduce noise / artifacts at cost of resolution.  '
                'Beware that abusing this can cause convergence issues.',
            )

        inference_group = 'Inference'
        inference_settings = self._model.inference_settings
        builder.add_spin_box(
            inference_settings.n_nearest_neighbors, 'Number of Neighbors:', group=inference_group
        )
        builder.add_spin_box(
            inference_settings.n_samples, 'Number of Samples:', group=inference_group
        )

        training_group = 'Training'
        training_settings = self._model.training_settings
        builder.add_view_controller(
            PowerTwoSpinBoxParameterViewController(training_settings.batch_size),
            'Batch Size:',
            group=training_group,
        )
        builder.add_spin_box(training_settings.nepochs, 'Number of Epochs:', group=training_group)

        if is_pinn:
            builder.add_decimal_slider(
                training_settings.mae_weight, 'Weight for MAE loss:', group=training_group
            )
            builder.add_decimal_slider(
                training_settings.nll_weight, 'Weight for NLL loss:', group=training_group
            )
            builder.add_decimal_slider(
                training_settings.realspace_mae_weight,
                'Realspace MAE Weight:',
                group=training_group,
            )
            builder.add_decimal_slider(
                training_settings.realspace_weight, 'Realspace Weight:', group=training_group
            )
            builder.add_check_box(
                training_settings.positions_provided,
                'Positions Provided',
                group=training_group,
            )
            builder.add_check_box(
                training_settings.probe_trainable,
                'Probe Trainable',
                group=training_group,
                tool_tip='Optimizes the probe function during training. Experimental feature.',
            )
            builder.add_check_box(
                training_settings.intensity_scale_trainable,
                'Intensity Scale Trainable',
                group=training_group,
                tool_tip="Optimize the model's internal amplitude scaling factor during training. "
                'Typically left True.',
            )

        return builder.build_widget()

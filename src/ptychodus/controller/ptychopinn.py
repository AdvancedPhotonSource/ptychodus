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

    def stepBy(self, steps: int) -> None:
        if steps < 0:
            self.setValue(self.value() // (1 << -steps))
        elif steps > 0:
            self.setValue(self.value() * (1 << steps))

    def validate(self, input_text: str, pos: int) -> tuple[QValidator.State, str, int]:
        try:
            value = int(input_text)
        except ValueError:
            pass
        else:
            if value > 0:
                is_pow2 = (value & (value - 1)) == 0

                if is_pow2:
                    return QValidator.Acceptable, input_text, pos

        return QValidator.Intermediate, input_text, pos


class PowerTwoSpinBoxParameterViewController(ParameterViewController, Observer):
    def __init__(self, parameter: IntegerParameter, *, tool_tip: str = '') -> None:
        super().__init__()
        self._parameter = parameter
        self._widget = PowerTwoSpinBox()

        if tool_tip:
            self._widget.setToolTip(tool_tip)

        self._syncModelToView()
        self._widget.valueChanged.connect(parameter.setValue)
        parameter.addObserver(self)

    def getWidget(self) -> QWidget:
        return self._widget

    def _syncModelToView(self) -> None:
        minimum = self._parameter.getMinimum()
        maximum = self._parameter.getMaximum()

        if minimum is None:
            raise ValueError('Minimum not provided!')

        if maximum is None:
            raise ValueError('Maximum not provided!')

        self._widget.blockSignals(True)
        self._widget.setRange(minimum, maximum)
        self._widget.setValue(self._parameter.getValue())
        self._widget.blockSignals(False)

    def update(self, observable: Observable) -> None:
        if observable is self._parameter:
            self._syncModelToView()


class PtychoPINNViewControllerFactory(ReconstructorViewControllerFactory):
    def __init__(
        self, model: PtychoPINNReconstructorLibrary, fileDialogFactory: FileDialogFactory
    ) -> None:
        super().__init__()
        self._model = model
        self._fileDialogFactory = fileDialogFactory

    @property
    def backendName(self) -> str:
        return 'PtychoPINN'

    def createViewController(self, reconstructorName: str) -> QWidget:
        builder = ParameterViewBuilder(self._fileDialogFactory)
        enumerators = self._model.enumerators

        model_group = 'Model'
        model_settings = self._model.model_settings
        builder.addSpinBox(
            model_settings.gridsize,
            'Grid Size for Model:',
            tool_tip='Controls number of images processed per solution region (e.g., gridsize=2 means 2^2=4 images at a time)',
            group=model_group,
        )
        builder.addSpinBox(
            model_settings.n_filters_scale,
            'Num. Filters Scale Factor:',
            tool_tip='Scale factor for number of filters',
            group=model_group,
        )
        builder.addComboBox(
            model_settings.amp_activation,
            enumerators.get_amp_activations(),
            'Amplitude Activation Function:',
            group=model_group,
        )
        builder.addCheckBox(
            model_settings.object_big,
            'Object Big',
            group=model_group,
            tool_tip='Enables a separate real-space reconstruction for each input diffraction image '
            'and an averaging / overlap constraint step. If False, no explicit averaging is performed '
            'and the decoders return a single real space image instead of `gridsize**2` images. '
            'Typically left True.',
        )
        builder.addCheckBox(
            model_settings.probe_big,
            'Probe Big',
            group=model_group,
            tool_tip='If True, enables a low-resolution reconstruction of the outer region of the NxN real-space grid. '
            'This technically violates the zero-padding / oversampling condition, '
            'but may be needed if the probe illumination has wide tails. '
            'Has no effect unless pad_object is True.',
        )
        builder.addCheckBox(
            model_settings.probe_mask,
            'Probe Mask',
            group=model_group,
            tool_tip='Whether to apply circular mask to the probe function. '
            "If toggling this changes the reconstruction, it's likely that there are edge / real space truncation artifacts. "
            'Should be used with pad_object = False.',
        )
        builder.addCheckBox(
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
        builder.addDecimalLineEdit(
            model_settings.probe_scale,
            'Probe Scale Factor:',
            group=model_group,
            tool_tip='Scaling factor for the probe amplitude. ',
        )
        builder.addDecimalLineEdit(
            model_settings.gaussian_smoothing_sigma,
            'Gaussian Smoothing Sigma:',
            group=model_group,
            tool_tip='Standard deviation for Gaussian smoothing of probe illumination.  '
            'Increase from 0 to reduce noise / artifacts at cost of resolution.  '
            'Beware that abusing this can cause convergence issues.',
        )

        training_group = 'Training'
        training_settings = self._model.training_settings
        builder.addFileOpener(
            training_settings.train_data_file,
            'Train Data File:',
            # FIXME parameters
            group=training_group,
        )
        builder.addFileOpener(
            training_settings.test_data_file,
            'Test Data File:',
            # FIXME parameters
            group=training_group,
        )
        builder.addViewController(
            PowerTwoSpinBoxParameterViewController(training_settings.batch_size),
            'Batch Size:',
            group=training_group,
        )
        builder.addSpinBox(training_settings.nepochs, 'Number of Epochs:', group=training_group)
        builder.addDecimalSlider(
            training_settings.mae_weight, 'Weight for MAE loss:', group=training_group
        )
        builder.addDecimalSlider(
            training_settings.nll_weight, 'Weight for NLL loss:', group=training_group
        )
        builder.addDecimalSlider(
            training_settings.realspace_mae_weight, 'Realspace MAE Weight:', group=training_group
        )
        builder.addDecimalSlider(
            training_settings.realspace_weight, 'Realspace Weight:', group=training_group
        )
        builder.addCheckBox(
            training_settings.positions_provided,
            'Positions Provided',
            group=training_group,
        )
        builder.addCheckBox(
            training_settings.probe_trainable,
            'Probe Trainable',
            group=training_group,
            tool_tip='Optimizes the probe function during training. Experimental feature.',
        )
        builder.addCheckBox(
            training_settings.intensity_scale_trainable,
            'Intensity Scale Trainable',
            group=training_group,
            tool_tip="Optimize the model's internal amplitude scaling factor during training. "
            'Typically left True.',
        )
        builder.addDirectoryChooser(
            training_settings.output_directory, 'Output Directory:', group=training_group
        )

        inference_group = 'Inference'
        inference_settings = self._model.inference_settings
        builder.addDirectoryChooser(
            inference_settings.output_directory, 'Output Directory:', group=inference_group
        )

        return builder.buildWidget()

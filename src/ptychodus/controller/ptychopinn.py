from PyQt5.QtWidgets import QWidget

from ..model.ptychopinn.core import PtychoPINNReconstructorLibrary
from .data import FileDialogFactory
from .parametric import ParameterViewBuilder
from .reconstructor import ReconstructorViewControllerFactory


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
        builder = ParameterViewBuilder()
        enumerators = self._model.enumerators

        model_group = 'Model'
        model_settings = self._model.model_settings
        builder.addSpinBox(model_settings.N, 'N:', group=model_group)  # FIXME 64, 128, 256
        builder.addIntegerLineEdit(model_settings.gridsize, 'Grid Size:', group=model_group)
        builder.addIntegerLineEdit(
            model_settings.n_filters_scale, 'N Filters Scale:', group=model_group
        )
        builder.addComboBox(
            model_settings.model_type,
            enumerators.get_model_types(),
            'Model Type:',
            group=model_group,
        )
        builder.addComboBox(
            model_settings.amp_activation,
            enumerators.get_amp_activations(),
            'Amp Activation:',
            group=model_group,
        )
        builder.addCheckBox(model_settings.object_big, 'Object Big', group=model_group)
        builder.addCheckBox(model_settings.probe_big, 'Probe Big', group=model_group)
        builder.addCheckBox(model_settings.probe_mask, 'Probe Mask', group=model_group)
        builder.addCheckBox(model_settings.pad_object, 'Pad Object', group=model_group)
        builder.addDecimalLineEdit(model_settings.probe_scale, 'Probe Scale:', group=model_group)

        training_group = 'Training'
        training_settings = self._model.training_settings
        builder.addFileChooser(
            training_settings.train_data_file, 'Train Data File:', group=training_group
        )
        builder.addFileChooser(
            training_settings.test_data_file, 'Test Data File:', group=training_group
        )
        builder.addSpinBox(
            training_settings.batch_size, 'Batch Size:', group=training_group
        )  # FIXME must be positive powers of two
        builder.addSpinBox(training_settings.nepochs, 'Number of Epochs:', group=training_group)
        builder.addDecimalSlider(training_settings.mae_weight, 'MAE Weight:', group=training_group)
        builder.addDecimalSlider(training_settings.nll_weight, 'NLL Weight:', group=training_group)
        builder.addDecimalSlider(
            training_settings.realspace_mae_weight, 'Realspace MAE Weight:', group=training_group
        )
        builder.addDecimalSlider(
            training_settings.realspace_weight, 'Realspace Weight:', group=training_group
        )
        builder.addComboBox(
            training_settings.data_source,
            enumerators.get_data_sources(),
            'Data Source:',
            group=training_group,
        )
        builder.addCheckBox(
            training_settings.probe_trainable, 'Probe Trainable', group=training_group
        )
        builder.addCheckBox(
            training_settings.intensity_scale_trainable,
            'Intensity Scale Trainable',
            group=training_group,
        )
        builder.addDirectoryChooser(
            training_settings.output_dir, 'Output Directory:', group=training_group
        )

        inference_group = 'Inference'
        inference_settings = self._model.inference_settings
        builder.addFileChooser(inference_settings.model_path, 'Model Path:', group=inference_group)
        builder.addDecimalLineEdit(
            inference_settings.gaussian_smoothing_sigma,
            'Gaussian Smoothing Sigma:',
            group=inference_group,
        )
        builder.addDirectoryChooser(
            inference_settings.output_dir, 'Output Directory:', group=inference_group
        )

        return builder.buildWidget()

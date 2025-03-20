from PyQt5.QtWidgets import QWidget

from ..model.ptychonn import PtychoNNReconstructorLibrary
from .parametric import ParameterViewBuilder
from .reconstructor import ReconstructorViewControllerFactory


class PtychoNNViewControllerFactory(ReconstructorViewControllerFactory):
    def __init__(self, model: PtychoNNReconstructorLibrary) -> None:
        super().__init__()
        self._model = model

    @property
    def backend_name(self) -> str:
        return 'PtychoNN'

    def create_view_controller(self, reconstructor_name: str) -> QWidget:
        view_builder = ParameterViewBuilder()

        model_settings = self._model.model_settings
        model_group = 'Model Parameters'
        view_builder.add_spin_box(
            model_settings.num_convolution_kernels, 'Convolution Kernels:', group=model_group
        )
        view_builder.add_spin_box(model_settings.batch_size, 'Batch Size:', group=model_group)
        view_builder.add_check_box(
            model_settings.use_batch_normalization, 'Use Batch Normalization', group=model_group
        )

        training_settings = self._model.training_settings
        training_group = 'Training Parameters'

        view_builder.add_decimal_slider(
            training_settings.validation_set_fractional_size,
            'Validation Set Fractional Size:',
            group=training_group,
        )
        view_builder.add_decimal_line_edit(
            training_settings.max_learning_rate, 'Max Learning Rate:', group=training_group
        )
        view_builder.add_decimal_line_edit(
            training_settings.min_learning_rate, 'Min Learning Rate:', group=training_group
        )
        view_builder.add_spin_box(
            training_settings.training_epochs, 'Training Epochs:', group=training_group
        )
        view_builder.add_spin_box(
            training_settings.status_interval_in_epochs, 'Status Interval:', group=training_group
        )

        return view_builder.build_widget()

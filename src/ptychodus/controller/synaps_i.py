from PyQt5.QtWidgets import QWidget

from ..model.synaps_i import SynapsIReconstructorLibrary
from .data import FileDialogFactory
from .parametric import ParameterViewBuilder
from .reconstructor import ReconstructorViewControllerFactory


class SynapsIViewControllerFactory(ReconstructorViewControllerFactory):
    def __init__(
        self, model: SynapsIReconstructorLibrary, file_dialog_factory: FileDialogFactory
    ) -> None:
        super().__init__()
        self._model = model
        self._file_dialog_factory = file_dialog_factory

    @property
    def backend_name(self) -> str:
        return 'SYNAPS-I'

    def create_view_controller(self, reconstructor_name: str) -> QWidget:
        builder = ParameterViewBuilder(self._file_dialog_factory)
        inference_settings = self._model.inference_settings

        inference_group = 'Inference'
        builder.add_file_opener(
            inference_settings.config_path,
            'Config YAML:',
            caption='Open SYNAPS-I Config',
            name_filters=[self._model_filter()],
            selected_name_filter=self._model_filter(),
            group=inference_group,
        )
        builder.add_spin_box(
            inference_settings.batch_size,
            'Batch Size:',
            group=inference_group,
        )
        builder.add_check_box(
            inference_settings.use_cuda,
            'Use CUDA',
            group=inference_group,
        )
        builder.add_spin_box(
            inference_settings.max_probe_modes,
            'Max Probe Modes:',
            group=inference_group,
        )
        builder.add_decimal_line_edit(
            inference_settings.normalization,
            'Normalization:',
            group=inference_group,
        )
        builder.add_file_opener(
            inference_settings.normalization_dict_path,
            'Normalization Dict:',
            caption='Open Normalization Dict',
            name_filters=['Pickle Files (*.pkl)'],
            selected_name_filter='Pickle Files (*.pkl)',
            group=inference_group,
        )
        builder.add_decimal_line_edit(
            inference_settings.scale,
            'Scale:',
            group=inference_group,
        )

        return builder.build_widget()

    @staticmethod
    def _model_filter() -> str:
        return 'YAML Files (*.yaml *.yml)'

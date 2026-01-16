from decimal import Decimal

from PyQt5.QtWidgets import QWidget

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.parametric import BooleanParameter, RealParameter

from ..model.synaps_i import SynapsIReconstructorLibrary
from .data import FileDialogFactory
from .parametric import ParameterViewBuilder, ParameterViewController
from .reconstructor import ReconstructorViewControllerFactory
from ..view.widgets import DecimalLineEdit


class NormalizationValueController(ParameterViewController, Observer):
    def __init__(
        self,
        normalization: RealParameter,
        specify_normalization: BooleanParameter,
        *,
        tool_tip: str = '',
    ) -> None:
        super().__init__()
        self._normalization = normalization
        self._specify_normalization = specify_normalization
        self._widget = DecimalLineEdit.create_instance(is_signed=False)

        if tool_tip:
            self._widget.setToolTip(tool_tip)

        self._sync_model_to_view()
        self._widget.value_changed.connect(self._sync_view_to_model)
        normalization.add_observer(self)
        specify_normalization.add_observer(self)

    def get_widget(self) -> QWidget:
        return self._widget

    def _sync_view_to_model(self, value: Decimal) -> None:
        self._normalization.set_value(float(value))

    def _sync_model_to_view(self) -> None:
        self._widget.set_value(Decimal(str(self._normalization.get_value())))
        self._widget.setEnabled(self._specify_normalization.get_value())

    def _update(self, observable: Observable) -> None:
        if observable in (self._normalization, self._specify_normalization):
            self._sync_model_to_view()


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
        builder.add_check_box(
            inference_settings.specify_normalization,
            'Specify Normalization',
            group=inference_group,
        )
        builder.add_view_controller(
            NormalizationValueController(
                inference_settings.normalization,
                inference_settings.specify_normalization,
            ),
            'Normalization:',
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

from decimal import Decimal
from typing import Any, Final
import logging

from PyQt5.QtCore import Qt, QAbstractListModel, QModelIndex, QObject, QStringListModel
from PyQt5.QtWidgets import QWidget

from ptychodus.api.observer import Observable, Observer

from ...model.fluorescence import (
    FluorescenceEnhancer,
    TwoStepFluorescenceEnhancingAlgorithm,
    VSPIFluorescenceEnhancingAlgorithm,
)
from ...model.visualization import VisualizationEngine
from ...view.probe import (
    FluorescenceDialog,
    FluorescenceTwoStepParametersView,
    FluorescenceVSPIParametersView,
)
from ...view.widgets import ExceptionDialog
from ..data import FileDialogFactory
from ..helpers import connect_current_changed_signal
from ..visualization import (
    VisualizationParametersController,
    VisualizationWidgetController,
)

logger = logging.getLogger(__name__)


class FluorescenceChannelListModel(QAbstractListModel):
    def __init__(self, enhancer: FluorescenceEnhancer, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._enhancer = enhancer

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        # TODO make this a table model and show measured/enhanced count statistics
        if index.isValid() and role == Qt.ItemDataRole.DisplayRole:
            emap = self._enhancer.get_measured_element_map(index.row())
            return emap.name

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return self._enhancer.get_num_channels()


class FluorescenceTwoStepViewController(Observer):
    def __init__(self, algorithm: TwoStepFluorescenceEnhancingAlgorithm) -> None:
        super().__init__()
        self._algorithm = algorithm
        self._view = FluorescenceTwoStepParametersView()

        self._upscaling_model = QStringListModel()
        self._upscaling_model.setStringList(self._algorithm.get_upscaling_strategies())
        self._view.upscaling_strategy_combo_box.setModel(self._upscaling_model)
        self._view.upscaling_strategy_combo_box.textActivated.connect(
            algorithm.set_upscaling_strategy
        )

        self._deconvolution_model = QStringListModel()
        self._deconvolution_model.setStringList(self._algorithm.get_deconvolution_strategies())
        self._view.deconvolution_strategy_combo_box.setModel(self._deconvolution_model)
        self._view.deconvolution_strategy_combo_box.textActivated.connect(
            algorithm.set_deconvolution_strategy
        )

        self._sync_model_to_view()
        algorithm.add_observer(self)

    def get_widget(self) -> QWidget:
        return self._view

    def _sync_model_to_view(self) -> None:
        self._view.upscaling_strategy_combo_box.setCurrentText(
            self._algorithm.get_upscaling_strategy()
        )
        self._view.deconvolution_strategy_combo_box.setCurrentText(
            self._algorithm.get_deconvolution_strategy()
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._algorithm:
            self._sync_model_to_view()


class FluorescenceVSPIViewController(Observer):
    MAX_INT: Final[int] = 0x7FFFFFFF

    def __init__(self, algorithm: VSPIFluorescenceEnhancingAlgorithm) -> None:
        super().__init__()
        self._algorithm = algorithm
        self._view = FluorescenceVSPIParametersView()

        self._view.damping_factor_line_edit.value_changed.connect(
            self._sync_damping_factor_to_model
        )
        self._view.max_iterations_spin_box.setRange(1, self.MAX_INT)
        self._view.max_iterations_spin_box.valueChanged.connect(algorithm.set_max_iterations)

        algorithm.add_observer(self)
        self._sync_model_to_view()

    def get_widget(self) -> QWidget:
        return self._view

    def _sync_damping_factor_to_model(self, value: Decimal) -> None:
        self._algorithm.set_damping_factor(float(value))

    def _sync_model_to_view(self) -> None:
        self._view.damping_factor_line_edit.set_value(
            Decimal(repr(self._algorithm.get_damping_factor()))
        )
        self._view.max_iterations_spin_box.setValue(self._algorithm.get_max_iterations())

    def _update(self, observable: Observable) -> None:
        if observable is self._algorithm:
            self._sync_model_to_view()


class FluorescenceViewController(Observer):
    def __init__(
        self,
        enhancer: FluorescenceEnhancer,
        engine: VisualizationEngine,
        file_dialog_factory: FileDialogFactory,
    ) -> None:
        super().__init__()
        self._enhancer = enhancer
        self._engine = engine
        self._file_dialog_factory = file_dialog_factory
        self._dialog = FluorescenceDialog()
        self._enhancement_model = QStringListModel()
        self._enhancement_model.setStringList(self._enhancer.algorithms())
        self._channel_list_model = FluorescenceChannelListModel(enhancer)

        self._dialog.fluorescence_parameters_view.open_button.clicked.connect(
            self._open_measured_dataset
        )

        two_step_view_controller = FluorescenceTwoStepViewController(
            enhancer.two_step_enhancing_algorithm
        )
        self._dialog.fluorescence_parameters_view.algorithm_combo_box.addItem(
            TwoStepFluorescenceEnhancingAlgorithm.DISPLAY_NAME,
            self._dialog.fluorescence_parameters_view.algorithm_combo_box.count(),
        )
        self._dialog.fluorescence_parameters_view.stacked_widget.addWidget(
            two_step_view_controller.get_widget()
        )

        vspi_view_controller = FluorescenceVSPIViewController(enhancer.vspi_enhancing_algorithm)
        self._dialog.fluorescence_parameters_view.algorithm_combo_box.addItem(
            VSPIFluorescenceEnhancingAlgorithm.DISPLAY_NAME,
            self._dialog.fluorescence_parameters_view.algorithm_combo_box.count(),
        )
        self._dialog.fluorescence_parameters_view.stacked_widget.addWidget(
            vspi_view_controller.get_widget()
        )

        self._dialog.fluorescence_parameters_view.algorithm_combo_box.textActivated.connect(
            enhancer.set_algorithm
        )
        self._dialog.fluorescence_parameters_view.algorithm_combo_box.currentIndexChanged.connect(
            self._dialog.fluorescence_parameters_view.stacked_widget.setCurrentIndex
        )
        self._dialog.fluorescence_parameters_view.algorithm_combo_box.setModel(
            self._enhancement_model
        )
        self._dialog.fluorescence_parameters_view.algorithm_combo_box.textActivated.connect(
            enhancer.set_algorithm
        )

        self._dialog.fluorescence_parameters_view.enhance_button.clicked.connect(
            self._enhance_fluorescence
        )
        self._dialog.fluorescence_parameters_view.save_button.clicked.connect(
            self._save_enhanced_dataset
        )

        self._dialog.fluorescence_channel_list_view.setModel(self._channel_list_model)
        connect_current_changed_signal(
            self._dialog.fluorescence_channel_list_view, self._update_view
        )

        self._measured_widget_controller = VisualizationWidgetController(
            engine,
            self._dialog.measured_widget,
            self._dialog.status_bar,
            file_dialog_factory,
        )
        self._enhanced_widget_controller = VisualizationWidgetController(
            engine,
            self._dialog.enhanced_widget,
            self._dialog.status_bar,
            file_dialog_factory,
        )
        self._visualization_parameters_controller = (
            VisualizationParametersController.create_instance(
                engine, self._dialog.visualization_parameters_view
            )
        )

        enhancer.add_observer(self)

    def _open_measured_dataset(self) -> None:
        title = 'Open Measured Fluorescence Dataset'
        file_path, name_filter = self._file_dialog_factory.get_open_file_path(
            self._dialog,
            title,
            name_filters=[nf for nf in self._enhancer.get_open_file_filters()],
            selected_name_filter=self._enhancer.get_open_file_filter(),
        )

        if file_path:
            try:
                self._enhancer.open_measured_dataset(file_path, name_filter)
            except Exception as err:
                logger.exception(err)
                ExceptionDialog.show_exception(title, err)

    def _enhance_fluorescence(self) -> None:
        try:
            self._enhancer.enhance_fluorescence()
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.show_exception('Enhance Fluorescence', err)

    def launch(self, product_index: int) -> None:
        self._enhancer.set_product(product_index)

        try:
            item_name = self._enhancer.get_product_name()
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.show_exception('Launch', err)
        else:
            self._dialog.setWindowTitle(f'Enhance Fluorescence: {item_name}')
            self._dialog.open()

    def _save_enhanced_dataset(self) -> None:
        title = 'Save Enhanced Fluorescence Dataset'
        file_path, name_filter = self._file_dialog_factory.get_save_file_path(
            self._dialog,
            title,
            name_filters=[nf for nf in self._enhancer.get_save_file_filters()],
            selected_name_filter=self._enhancer.get_save_file_filter(),
        )

        if file_path:
            try:
                self._enhancer.save_enhanced_dataset(file_path, name_filter)
            except Exception as err:
                logger.exception(err)
                ExceptionDialog.show_exception(title, err)

    def _sync_model_to_view(self) -> None:
        self._dialog.fluorescence_parameters_view.algorithm_combo_box.setCurrentText(
            self._enhancer.get_algorithm()
        )
        self._channel_list_model.beginResetModel()
        self._channel_list_model.endResetModel()

    def _update_view(self, current: QModelIndex, previous: QModelIndex) -> None:
        if not current.isValid():
            self._measured_widget_controller.clear_array()
            self._enhanced_widget_controller.clear_array()
            return

        try:
            emap_measured = self._enhancer.get_measured_element_map(current.row())
        except Exception as err:
            logger.exception(err)
            self._measured_widget_controller.clear_array()
            ExceptionDialog.show_exception('Render Measured Element Map', err)
        else:
            self._measured_widget_controller.set_array(
                emap_measured.counts_per_second, self._enhancer.get_pixel_geometry()
            )

        try:
            emap_enhanced = self._enhancer.get_enhanced_element_map(current.row())
        except Exception as err:
            logger.exception(err)
            self._enhanced_widget_controller.clear_array()
            ExceptionDialog.show_exception('Render Enhanced Element Map', err)
        else:
            self._enhanced_widget_controller.set_array(
                emap_enhanced.counts_per_second, self._enhancer.get_pixel_geometry()
            )

    def _update(self, observable: Observable) -> None:
        if observable is self._enhancer:
            self._sync_model_to_view()

from decimal import Decimal
from typing import Any, Final
import logging

from PyQt5.QtCore import Qt, QAbstractListModel, QModelIndex, QObject, QStringListModel
from PyQt5.QtWidgets import QWidget

from ptychodus.api.fluorescence import ElementMap, FluorescenceDataset
from ptychodus.api.observer import Observable, Observer
from ptychodus.api.plugins import PluginChooser

from ...model.fluorescence import (
    FluorescenceAPI,
    FluorescenceTaskMonitor,
    PtychozoonFluorescenceEnhancer,
    TwoStepFluorescenceEnhancer,
    VSPIFluorescenceEnhancer,
)
from ...model.visualization import VisualizationEngine
from ...view.probe import (
    FluorescenceDialog,
    FluorescencePtychozoonParametersView,
    FluorescenceStatusView,
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
    def __init__(
        self, controller: 'FluorescenceViewController', parent: QObject | None = None
    ) -> None:
        super().__init__(parent)
        self._controller = controller

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        # TODO make this a table model and show measured/enhanced count statistics
        if index.isValid() and role == Qt.ItemDataRole.DisplayRole:
            emap = self._controller.get_measured_element_map(index.row())

            if emap is not None:
                return emap.name

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        return self._controller.get_num_channels()


class FluorescenceTwoStepViewController(Observer):
    def __init__(self, enhancer: TwoStepFluorescenceEnhancer) -> None:
        super().__init__()
        self._enhancer = enhancer
        self._view = FluorescenceTwoStepParametersView()

        self._upscaling_model = QStringListModel()
        self._upscaling_model.setStringList(self._enhancer.get_upscaling_strategies())
        self._view.upscaling_strategy_combo_box.setModel(self._upscaling_model)
        self._view.upscaling_strategy_combo_box.textActivated.connect(
            enhancer.set_upscaling_strategy
        )

        self._deconvolution_model = QStringListModel()
        self._deconvolution_model.setStringList(self._enhancer.get_deconvolution_strategies())
        self._view.deconvolution_strategy_combo_box.setModel(self._deconvolution_model)
        self._view.deconvolution_strategy_combo_box.textActivated.connect(
            enhancer.set_deconvolution_strategy
        )

        self._sync_model_to_view()
        enhancer.add_observer(self)

    def get_widget(self) -> QWidget:
        return self._view

    def _sync_model_to_view(self) -> None:
        self._view.upscaling_strategy_combo_box.setCurrentText(
            self._enhancer.get_upscaling_strategy()
        )
        self._view.deconvolution_strategy_combo_box.setCurrentText(
            self._enhancer.get_deconvolution_strategy()
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._enhancer:
            self._sync_model_to_view()


class FluorescenceVSPIViewController(Observer):
    MAX_INT: Final[int] = 0x7FFFFFFF

    def __init__(self, enhancer: VSPIFluorescenceEnhancer) -> None:
        super().__init__()
        self._enhancer = enhancer
        self._view = FluorescenceVSPIParametersView()

        self._view.damping_factor_line_edit.value_changed.connect(
            self._sync_damping_factor_to_model
        )
        self._view.max_iterations_spin_box.setRange(1, self.MAX_INT)
        self._view.max_iterations_spin_box.valueChanged.connect(enhancer.set_max_iterations)

        enhancer.add_observer(self)
        self._sync_model_to_view()

    def get_widget(self) -> QWidget:
        return self._view

    def _sync_damping_factor_to_model(self, value: Decimal) -> None:
        self._enhancer.set_damping_factor(float(value))

    def _sync_model_to_view(self) -> None:
        self._view.damping_factor_line_edit.set_value(
            Decimal(repr(self._enhancer.get_damping_factor()))
        )
        self._view.max_iterations_spin_box.setValue(self._enhancer.get_max_iterations())

    def _update(self, observable: Observable) -> None:
        if observable is self._enhancer:
            self._sync_model_to_view()


class FluorescencePtychozoonViewController(Observer):
    MAX_INT: Final[int] = 0x7FFFFFFF

    def __init__(self, enhancer: PtychozoonFluorescenceEnhancer) -> None:
        super().__init__()
        self._enhancer = enhancer
        self._view = FluorescencePtychozoonParametersView()

        self._view.damping_factor_line_edit.value_changed.connect(
            self._sync_damping_factor_to_model
        )
        self._view.gradient_smoothness_line_edit.value_changed.connect(
            self._sync_gradient_smoothness_to_model
        )
        self._view.max_iterations_spin_box.setRange(1, self.MAX_INT)
        self._view.max_iterations_spin_box.valueChanged.connect(enhancer.set_max_iterations)
        self._view.atol_line_edit.value_changed.connect(self._sync_atol_to_model)
        self._view.btol_line_edit.value_changed.connect(self._sync_btol_to_model)
        self._view.checkpoint_interval_spin_box.setRange(1, self.MAX_INT)
        self._view.checkpoint_interval_spin_box.valueChanged.connect(
            enhancer.set_checkpoint_interval
        )
        self._view.use_gpu_check_box.toggled.connect(enhancer.set_gpu_enabled)
        self._view.gpu_device_index_spin_box.setRange(0, self.MAX_INT)
        self._view.gpu_device_index_spin_box.valueChanged.connect(enhancer.set_gpu_device_index)

        enhancer.add_observer(self)
        self._sync_model_to_view()

    def get_widget(self) -> QWidget:
        return self._view

    def _sync_damping_factor_to_model(self, value: Decimal) -> None:
        self._enhancer.set_damping_factor(float(value))

    def _sync_gradient_smoothness_to_model(self, value: Decimal) -> None:
        self._enhancer.set_gradient_smoothness(float(value))

    def _sync_atol_to_model(self, value: Decimal) -> None:
        self._enhancer.set_atol(float(value))

    def _sync_btol_to_model(self, value: Decimal) -> None:
        self._enhancer.set_btol(float(value))

    def _sync_model_to_view(self) -> None:
        self._view.damping_factor_line_edit.set_value(
            Decimal(repr(self._enhancer.get_damping_factor()))
        )
        self._view.gradient_smoothness_line_edit.set_value(
            Decimal(repr(self._enhancer.get_gradient_smoothness()))
        )
        self._view.max_iterations_spin_box.setValue(self._enhancer.get_max_iterations())
        self._view.atol_line_edit.set_value(Decimal(repr(self._enhancer.get_atol())))
        self._view.btol_line_edit.set_value(Decimal(repr(self._enhancer.get_btol())))
        self._view.checkpoint_interval_spin_box.setValue(self._enhancer.get_checkpoint_interval())
        self._view.use_gpu_check_box.setChecked(self._enhancer.is_gpu_enabled())
        self._view.gpu_device_index_spin_box.setValue(self._enhancer.get_gpu_device_index())

    def _update(self, observable: Observable) -> None:
        if observable is self._enhancer:
            self._sync_model_to_view()


class FluorescenceStatusController(Observer):
    def __init__(
        self,
        task_monitor: FluorescenceTaskMonitor,
        view: FluorescenceStatusView,
    ) -> None:
        super().__init__()
        self._monitor = task_monitor
        self._view = view

        view.stop_button.clicked.connect(self._monitor.stop_processing)

        self._sync_model_to_view()
        self._monitor.add_observer(self)

    def _sync_model_to_view(self) -> None:
        log_handler = self._monitor.get_log_handler()

        for text in log_handler.messages():
            self._view.text_edit.appendPlainText(text)

        progress_goal = self._monitor.get_progress_goal()
        progress_bar = self._view.progress_bar

        if self._monitor.is_processing and progress_goal > 0:
            progress_bar.show()
            progress_bar.setRange(0, progress_goal)
            progress_bar.setValue(self._monitor.get_progress())
            self._view.stop_button.show()
        else:
            progress_bar.hide()
            self._view.stop_button.hide()

    def _update(self, observable: Observable) -> None:
        if observable is self._monitor:
            self._sync_model_to_view()


class FluorescenceViewController(Observer):
    def __init__(
        self,
        fluorescence_api: FluorescenceAPI,
        enhancer_chooser: PluginChooser,
        two_step_enhancer: TwoStepFluorescenceEnhancer,
        vspi_enhancer: VSPIFluorescenceEnhancer,
        ptychozoon_enhancer: PtychozoonFluorescenceEnhancer | None,
        task_monitor: FluorescenceTaskMonitor,
        engine: VisualizationEngine,
        file_dialog_factory: FileDialogFactory,
    ) -> None:
        super().__init__()
        self._fluorescence_api = fluorescence_api
        self._enhancer_chooser = enhancer_chooser
        self._dataset_emitter = task_monitor.get_dataset_emitter()
        self._engine = engine
        self._file_dialog_factory = file_dialog_factory
        self._dialog = FluorescenceDialog()
        self._product_index = -1
        self._measured: FluorescenceDataset | None = None
        self._enhanced: FluorescenceDataset | None = None
        self._status_controller = FluorescenceStatusController(
            task_monitor,
            self._dialog.fluorescence_status_view,
        )
        self._enhancement_model = QStringListModel()
        self._enhancement_model.setStringList([plugin.display_name for plugin in enhancer_chooser])
        self._channel_list_model = FluorescenceChannelListModel(self)

        self._dialog.fluorescence_parameters_view.open_button.clicked.connect(
            self._open_measured_dataset
        )

        two_step_view_controller = FluorescenceTwoStepViewController(two_step_enhancer)
        self._dialog.fluorescence_parameters_view.algorithm_combo_box.addItem(
            TwoStepFluorescenceEnhancer.DISPLAY_NAME,
            self._dialog.fluorescence_parameters_view.algorithm_combo_box.count(),
        )
        self._dialog.fluorescence_parameters_view.stacked_widget.addWidget(
            two_step_view_controller.get_widget()
        )

        vspi_view_controller = FluorescenceVSPIViewController(vspi_enhancer)
        self._dialog.fluorescence_parameters_view.algorithm_combo_box.addItem(
            VSPIFluorescenceEnhancer.DISPLAY_NAME,
            self._dialog.fluorescence_parameters_view.algorithm_combo_box.count(),
        )
        self._dialog.fluorescence_parameters_view.stacked_widget.addWidget(
            vspi_view_controller.get_widget()
        )

        # Registered last, matching the enhancer_chooser order, so the combo-box
        # index selects the correct stacked page. Only present when ptychozoon is
        # installed (enhancer is None otherwise).
        self._ptychozoon_view_controller: FluorescencePtychozoonViewController | None = None

        if ptychozoon_enhancer is not None:
            self._ptychozoon_view_controller = FluorescencePtychozoonViewController(
                ptychozoon_enhancer
            )
            self._dialog.fluorescence_parameters_view.algorithm_combo_box.addItem(
                PtychozoonFluorescenceEnhancer.DISPLAY_NAME,
                self._dialog.fluorescence_parameters_view.algorithm_combo_box.count(),
            )
            self._dialog.fluorescence_parameters_view.stacked_widget.addWidget(
                self._ptychozoon_view_controller.get_widget()
            )

        self._dialog.fluorescence_parameters_view.algorithm_combo_box.textActivated.connect(
            enhancer_chooser.set_current_plugin
        )
        self._dialog.fluorescence_parameters_view.algorithm_combo_box.currentIndexChanged.connect(
            self._dialog.fluorescence_parameters_view.stacked_widget.setCurrentIndex
        )
        self._dialog.fluorescence_parameters_view.algorithm_combo_box.setModel(
            self._enhancement_model
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
        self._visualization_parameters_controller = VisualizationParametersController(
            engine, self._dialog.visualization_parameters_view
        )

        enhancer_chooser.add_observer(self)
        self._dataset_emitter.add_observer(self)
        self._sync_algorithm_combo_box()

    def get_num_channels(self) -> int:
        return 0 if self._measured is None else len(self._measured.element_maps)

    def get_measured_element_map(self, channel_index: int) -> ElementMap | None:
        if self._measured is None:
            return None

        return self._measured.element_maps[channel_index]

    def get_enhanced_element_map(self, channel_index: int) -> ElementMap | None:
        if self._enhanced is not None:
            return self._enhanced.element_maps[channel_index]

        return self.get_measured_element_map(channel_index)

    def _reset_channel_list(self) -> None:
        self._channel_list_model.beginResetModel()
        self._channel_list_model.endResetModel()

    def _open_measured_dataset(self) -> None:
        title = 'Open Measured Fluorescence Dataset'
        file_path, name_filter = self._file_dialog_factory.get_open_file_path(
            self._dialog,
            title,
            name_filters=[nf for nf in self._fluorescence_api.get_open_file_filters()],
            selected_name_filter=self._fluorescence_api.get_open_file_filter(),
        )

        if file_path:
            try:
                self._measured = self._fluorescence_api.load_measured_dataset(
                    file_path, file_type=name_filter
                )
            except Exception as err:
                logger.exception(err)
                ExceptionDialog.show_exception(title, err)
            else:
                self._enhanced = None
                self._reset_channel_list()

    def _enhance_fluorescence(self) -> None:
        if self._measured is None:
            ExceptionDialog.show_exception(
                'Enhance Fluorescence', ValueError('Fluorescence dataset not loaded!')
            )
            return

        try:
            self._fluorescence_api.enhance(self._product_index, self._measured)
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.show_exception('Enhance Fluorescence', err)

    def launch(self, product_index: int) -> None:
        self._product_index = product_index
        self._measured = None
        self._enhanced = None
        self._reset_channel_list()

        try:
            item_name = self._fluorescence_api.get_product_name(product_index)
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.show_exception('Launch', err)
        else:
            self._dialog.setWindowTitle(f'Enhance Fluorescence: {item_name}')
            self._dialog.open()

    def _save_enhanced_dataset(self) -> None:
        title = 'Save Enhanced Fluorescence Dataset'

        if self._enhanced is None:
            ExceptionDialog.show_exception(title, ValueError('Fluorescence dataset not enhanced!'))
            return

        file_path, name_filter = self._file_dialog_factory.get_save_file_path(
            self._dialog,
            title,
            name_filters=[nf for nf in self._fluorescence_api.get_save_file_filters()],
            selected_name_filter=self._fluorescence_api.get_save_file_filter(),
        )

        if file_path:
            try:
                self._fluorescence_api.save_enhanced_dataset(
                    self._enhanced, file_path, file_type=name_filter
                )
            except Exception as err:
                logger.exception(err)
                ExceptionDialog.show_exception(title, err)

    def _sync_algorithm_combo_box(self) -> None:
        self._dialog.fluorescence_parameters_view.algorithm_combo_box.setCurrentText(
            self._enhancer_chooser.get_current_plugin().display_name
        )

    def _update_view(self, current: QModelIndex, previous: QModelIndex) -> None:
        if not current.isValid():
            self._measured_widget_controller.clear_array()
            self._enhanced_widget_controller.clear_array()
            return

        try:
            pixel_geometry = self._fluorescence_api.get_pixel_geometry(self._product_index)
        except Exception as err:
            logger.exception(err)
            self._measured_widget_controller.clear_array()
            self._enhanced_widget_controller.clear_array()
            ExceptionDialog.show_exception('Render Element Map', err)
            return

        emap_measured = self.get_measured_element_map(current.row())

        if emap_measured is None:
            self._measured_widget_controller.clear_array()
        else:
            self._measured_widget_controller.set_array(
                emap_measured.counts_per_second, pixel_geometry
            )

        emap_enhanced = self.get_enhanced_element_map(current.row())

        if emap_enhanced is None:
            self._enhanced_widget_controller.clear_array()
        else:
            self._enhanced_widget_controller.set_array(
                emap_enhanced.counts_per_second, pixel_geometry
            )

    def _update(self, observable: Observable) -> None:
        if observable is self._enhancer_chooser:
            self._sync_algorithm_combo_box()
        elif observable is self._dataset_emitter:
            self._enhanced = self._dataset_emitter.get_latest_enhanced()
            self._reset_channel_list()

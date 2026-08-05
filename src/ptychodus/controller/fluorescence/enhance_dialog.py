from __future__ import annotations
from decimal import Decimal
from typing import Final
import logging

from PyQt5.QtCore import QStringListModel
from PyQt5.QtWidgets import QWidget

from ptychodus.api.fluorescence import FluorescenceEnhancer
from ptychodus.api.observer import Observable, Observer
from ptychodus.api.plugins import PluginChooser

from ...model.fluorescence import (
    FluorescenceAPI,
    FluorescenceItemState,
    FluorescenceTaskMonitor,
    PtychozoonFluorescenceEnhancer,
    TwoStepFluorescenceEnhancer,
    VSPIFluorescenceEnhancer,
)
from ...view.fluorescence import (
    FluorescenceEnhanceDialog,
    FluorescencePtychozoonParametersView,
    FluorescenceStatusView,
    FluorescenceTwoStepParametersView,
    FluorescenceVSPIParametersView,
)
from ...view.widgets import ExceptionDialog

logger = logging.getLogger(__name__)


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


class FluorescenceEnhanceDialogController(Observer):
    """Owns the modal enhance dialog and its algorithm parameter sub-controllers.

    Launched from the top-level panel via ``launch(item_index)``; the target
    product is looked up through the fluorescence item itself (which holds a
    ProductRepositoryItem reference from load time). While the dialog is open,
    the Run button submits an enhancement task through FluorescenceAPI. The
    dialog stays open during enhancement so the user can watch the status log;
    the panel handles saving the enhanced result.
    """

    def __init__(
        self,
        fluorescence_api: FluorescenceAPI,
        enhancer_chooser: PluginChooser[FluorescenceEnhancer],
        two_step_enhancer: TwoStepFluorescenceEnhancer,
        vspi_enhancer: VSPIFluorescenceEnhancer,
        ptychozoon_enhancer: PtychozoonFluorescenceEnhancer | None,
        task_monitor: FluorescenceTaskMonitor,
    ) -> None:
        super().__init__()
        self._api = fluorescence_api
        self._enhancer_chooser = enhancer_chooser
        self._task_monitor = task_monitor
        self._dialog = FluorescenceEnhanceDialog()
        self._item_index = -1

        self._status_controller = FluorescenceStatusController(
            task_monitor, self._dialog.status_view
        )

        self._enhancement_model = QStringListModel()
        self._enhancement_model.setStringList([plugin.display_name for plugin in enhancer_chooser])

        parameters_view = self._dialog.parameters_view
        self._two_step_view_controller = FluorescenceTwoStepViewController(two_step_enhancer)
        parameters_view.algorithm_combo_box.addItem(
            TwoStepFluorescenceEnhancer.DISPLAY_NAME,
            parameters_view.algorithm_combo_box.count(),
        )
        parameters_view.stacked_widget.addWidget(self._two_step_view_controller.get_widget())

        self._vspi_view_controller = FluorescenceVSPIViewController(vspi_enhancer)
        parameters_view.algorithm_combo_box.addItem(
            VSPIFluorescenceEnhancer.DISPLAY_NAME,
            parameters_view.algorithm_combo_box.count(),
        )
        parameters_view.stacked_widget.addWidget(self._vspi_view_controller.get_widget())

        # Registered last, matching enhancer_chooser order so the combo-box
        # index selects the correct stacked page. Only present when ptychozoon
        # is installed (enhancer is None otherwise).
        self._ptychozoon_view_controller: FluorescencePtychozoonViewController | None = None
        if ptychozoon_enhancer is not None:
            self._ptychozoon_view_controller = FluorescencePtychozoonViewController(
                ptychozoon_enhancer
            )
            parameters_view.algorithm_combo_box.addItem(
                PtychozoonFluorescenceEnhancer.DISPLAY_NAME,
                parameters_view.algorithm_combo_box.count(),
            )
            parameters_view.stacked_widget.addWidget(self._ptychozoon_view_controller.get_widget())

        parameters_view.algorithm_combo_box.textActivated.connect(
            enhancer_chooser.set_current_plugin
        )
        parameters_view.algorithm_combo_box.currentIndexChanged.connect(
            parameters_view.stacked_widget.setCurrentIndex
        )
        parameters_view.algorithm_combo_box.setModel(self._enhancement_model)

        self._dialog.run_button.clicked.connect(self._run)

        enhancer_chooser.add_observer(self)
        task_monitor.add_observer(self)
        self._sync_algorithm_combo_box()
        self._sync_run_button_enabled()

    def launch(self, item_index: int) -> None:
        self._item_index = item_index
        try:
            item = self._api.get_item(item_index)
        except IndexError:
            logger.warning(f'Missing fluorescence item {item_index}')
            return
        product_name = item.get_product().get_name()
        self._dialog.setWindowTitle(
            f'Enhance Fluorescence "{item.get_label()}" against "{product_name}"'
        )
        self._sync_run_button_enabled()
        self._dialog.open()

    def _run(self) -> None:
        if self._item_index < 0:
            ExceptionDialog.show_exception(
                'Enhance Fluorescence',
                ValueError('Fluorescence item is not set'),
            )
            return
        try:
            self._api.enhance(self._item_index)
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.show_exception('Enhance Fluorescence', err)

    def _sync_algorithm_combo_box(self) -> None:
        self._dialog.parameters_view.algorithm_combo_box.setCurrentText(
            self._enhancer_chooser.get_current_plugin().display_name
        )

    def _sync_run_button_enabled(self) -> None:
        enabled = not self._task_monitor.is_processing and self._item_index >= 0
        if enabled:
            try:
                item = self._api.get_item(self._item_index)
            except IndexError:
                enabled = False
            else:
                if item.get_state() is not FluorescenceItemState.READY:
                    enabled = False
        self._dialog.run_button.setEnabled(enabled)

    def _update(self, observable: Observable) -> None:
        if observable is self._enhancer_chooser:
            self._sync_algorithm_combo_box()
        elif observable is self._task_monitor:
            self._sync_run_button_enabled()

from __future__ import annotations
import logging

from PyQt5.QtCore import QStringListModel
from PyQt5.QtWidgets import QWidget

from ptychodus.api.observer import Observable, Observer

from ...model.fluorescence import (
    FluorescenceCore,
    FluorescenceItemState,
    FluorescenceSettings,
    FluorescenceTaskMonitor,
    PtychozoonFluorescenceEnhancer,
    TwoStepFluorescenceEnhancer,
    VSPIFluorescenceEnhancer,
)
from ...view.fluorescence import (
    FluorescenceEnhanceDialog,
    FluorescenceStatusView,
)
from ...view.widgets import ExceptionDialog
from ..parametric import ParameterViewBuilder

logger = logging.getLogger(__name__)


def _build_two_step_widget(core: FluorescenceCore) -> QWidget:
    builder = ParameterViewBuilder()
    builder.add_combo_box(
        core.upscaling_strategy_parameter,
        [plugin.display_name for plugin in core.upscaling_strategy_chooser],
        'Upscaling Strategy:',
    )
    builder.add_combo_box(
        core.deconvolution_strategy_parameter,
        [plugin.display_name for plugin in core.deconvolution_strategy_chooser],
        'Deconvolution Strategy:',
    )
    return _build_page(builder)


def _build_vspi_widget(settings: FluorescenceSettings) -> QWidget:
    builder = ParameterViewBuilder()
    builder.add_decimal_line_edit(settings.vspi_damping_factor, 'Damping Factor:')
    builder.add_spin_box(settings.vspi_max_iterations, 'Max Iterations:')
    return _build_page(builder)


def _build_ptychozoon_widget(settings: FluorescenceSettings) -> QWidget:
    builder = ParameterViewBuilder()
    builder.add_decimal_line_edit(settings.ptychozoon_damping_factor, 'Damping Factor:')
    builder.add_decimal_line_edit(settings.ptychozoon_gradient_smoothness, 'Gradient Smoothness:')
    builder.add_spin_box(settings.ptychozoon_max_iterations, 'Max Iterations:')
    builder.add_decimal_line_edit(settings.ptychozoon_atol, 'A Tolerance:')
    builder.add_decimal_line_edit(settings.ptychozoon_btol, 'B Tolerance:')
    builder.add_spin_box(settings.ptychozoon_checkpoint_interval, 'Checkpoint Interval:')
    builder.add_check_box(settings.ptychozoon_use_gpu, 'Use GPU:')
    builder.add_spin_box(settings.ptychozoon_gpu_device_index, 'CUDA Device Index:')
    return _build_page(builder)


def _build_page(builder: ParameterViewBuilder) -> QWidget:
    """Build a stacked-widget page, matching the zero margins the old views used."""
    widget = builder.build_widget()
    layout = widget.layout()

    if layout is not None:
        layout.setContentsMargins(0, 0, 0, 0)

    return widget


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
        core: FluorescenceCore,
        *,
        has_ptychozoon: bool,
    ) -> None:
        super().__init__()
        enhancer_chooser = core.enhancer_chooser
        task_monitor = core.task_monitor
        self._api = core.fluorescence_api
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
        parameters_view.algorithm_combo_box.addItem(
            TwoStepFluorescenceEnhancer.DISPLAY_NAME,
            parameters_view.algorithm_combo_box.count(),
        )
        parameters_view.stacked_widget.addWidget(_build_two_step_widget(core))

        parameters_view.algorithm_combo_box.addItem(
            VSPIFluorescenceEnhancer.DISPLAY_NAME,
            parameters_view.algorithm_combo_box.count(),
        )
        parameters_view.stacked_widget.addWidget(_build_vspi_widget(core.settings))

        # Registered last, matching enhancer_chooser order so the combo-box
        # index selects the correct stacked page. Only present when ptychozoon
        # is installed.
        if has_ptychozoon:
            parameters_view.algorithm_combo_box.addItem(
                PtychozoonFluorescenceEnhancer.DISPLAY_NAME,
                parameters_view.algorithm_combo_box.count(),
            )
            parameters_view.stacked_widget.addWidget(_build_ptychozoon_widget(core.settings))

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

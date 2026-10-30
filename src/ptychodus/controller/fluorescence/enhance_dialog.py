from __future__ import annotations
import logging

from PyQt5.QtWidgets import QWidget

from ptychodus.model.observer import Observable, Observer

from ...model.fluorescence import (
    ENHANCER_LIBRARY,
    FluorescenceCore,
    FluorescenceItemState,
    FluorescenceTaskMonitor,
)
from ...view.fluorescence import (
    FluorescenceEnhanceDialog,
    FluorescenceStatusView,
)
from ...view.widgets import ExceptionDialog
from ..parameters import ComboBoxParameterViewController
from ..processing.generated import GeneratedViewControllerFactory

logger = logging.getLogger(__name__)


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

    def __init__(self, core: FluorescenceCore) -> None:
        super().__init__()
        task_monitor = core.task_monitor
        self._api = core.fluorescence_api
        self._enhancer_parameter = core.enhancer_parameter
        self._task_monitor = task_monitor
        self._algorithm_view_controller = ComboBoxParameterViewController(
            core.enhancer_parameter, core.enhancer_parameter.choices()
        )
        self._dialog = FluorescenceEnhanceDialog(self._algorithm_view_controller.get_widget())
        self._item_index = -1

        self._status_controller = FluorescenceStatusController(
            task_monitor, self._dialog.status_view
        )

        # One page per registered enhancer, laid out from what it declares by the
        # builder the reconstructor panels use. Keyed by name rather than by index:
        # the chooser sorts its plugins by display name, so page insertion order is
        # not the combo-box order.
        factory = GeneratedViewControllerFactory(ENHANCER_LIBRARY, core.enhancer_parameter_store)
        parameters_view = self._dialog.parameters_view
        self._pages: dict[str, QWidget] = {
            name: factory.create_view_controller(name) for name in core.enhancer_parameter.choices()
        }

        for page in self._pages.values():
            parameters_view.stacked_widget.addWidget(page)

        self._dialog.run_button.clicked.connect(self._run)

        core.enhancer_parameter.add_observer(self)
        task_monitor.add_observer(self)
        self._sync_algorithm_page()
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
            f'Enhance Fluorescence "{item.get_name()}" against "{product_name}"'
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

    def _sync_algorithm_page(self) -> None:
        page = self._pages.get(self._enhancer_parameter.get_value())

        if page is not None:
            self._dialog.parameters_view.stacked_widget.setCurrentWidget(page)

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
        if observable is self._enhancer_parameter:
            self._sync_algorithm_page()
        elif observable is self._task_monitor:
            self._sync_run_button_enabled()

from __future__ import annotations
from collections.abc import Sequence
import logging

from PyQt5.QtCore import QModelIndex
from PyQt5.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QHBoxLayout,
    QRadioButton,
    QWidget,
)

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.product import LossValue

from ...model.processing import ProcessingTaskMonitor
from ...model.product import ProductRepository, ProductRepositoryItem, ProductRepositoryObserver
from ...model.product.metadata import MetadataRepositoryItem
from ...model.product.object import ObjectRepositoryItem
from ...model.product.probe import ProbeRepositoryItem
from ...model.product.probe_positions import ProbePositionsRepositoryItem
from ...view.processing import ProcessingStatusView
from ..parametric import ParameterViewController
from ..product.core import ProductRepositoryComboProxyModel, ProductRepositoryTableModel

logger = logging.getLogger(__name__)


class ProcessingStatusController(Observer):
    def __init__(
        self,
        product_repository: ProductRepository,
        monitor: ProcessingTaskMonitor,
        view: ProcessingStatusView,
    ) -> None:
        super().__init__()
        self._product_repository = product_repository
        self._monitor = monitor
        self._view = view
        self._view.text_edit.setReadOnly(True)

        view.stop_button.clicked.connect(monitor.stop_processing)

        self._sync_model_to_view()
        monitor.add_observer(self)

    def plot_losses(self, product_index: int) -> None:
        # TODO fix so that plot and progress cannot mismatch
        if product_index < 0:
            self._view.axes.clear()
            return

        try:
            item = self._product_repository[product_index]
        except IndexError as exc:
            logger.exception(exc)
            return

        epoch = [loss.epoch for loss in item.get_losses()]
        losses = [loss.value for loss in item.get_losses()]

        ax = self._view.axes
        ax.clear()
        ax.set_title(item.get_name())
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.semilogy(epoch, losses, '.-', label='Loss', linewidth=2.0)
        ax.grid(True)
        self._view.figure_canvas.draw()

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


class ProductParameterViewController(ParameterViewController, ProductRepositoryObserver):
    """Combobox for choosing the product to reconstruct on.

    The Qt model (a proxy over the shared ProductRepositoryTableModel) keeps
    the combobox in sync with the repository automatically; the only reason
    this class still observes the repository is to refresh the loss plot when
    the currently-selected product finishes an epoch.
    """

    def __init__(
        self,
        repository: ProductRepository,
        product_table_model: ProductRepositoryTableModel,
        status_controller: ProcessingStatusController,
        *,
        tool_tip: str = '',
    ) -> None:
        super().__init__()
        self._status_controller = status_controller
        self._model = ProductRepositoryComboProxyModel(product_table_model, repository)
        self._widget = QComboBox()

        if tool_tip:
            self._widget.setToolTip(tool_tip)

        self._widget.setModel(self._model)
        self._widget.currentIndexChanged.connect(status_controller.plot_losses)

        self._model.rowsInserted.connect(lambda *_: self._auto_select_first_if_empty())
        self._model.rowsRemoved.connect(self._on_rows_removed)
        self._auto_select_first_if_empty()

        repository.add_observer(self)

    def get_widget(self) -> QComboBox:
        return self._widget

    def _auto_select_first_if_empty(self) -> None:
        if self._widget.currentIndex() >= 0:
            return
        if self._widget.count() > 0:
            self._widget.setCurrentIndex(0)

    def _on_rows_removed(self, parent: QModelIndex, first: int, last: int) -> None:
        if self._widget.currentIndex() >= 0:
            return
        row_count = self._widget.count()
        if row_count > 0:
            self._widget.setCurrentIndex(min(first, row_count - 1))

    def handle_item_inserted(self, index: int, item: ProductRepositoryItem) -> None:
        pass

    def handle_metadata_changed(self, index: int, item: MetadataRepositoryItem) -> None:
        pass

    def handle_probe_positions_changed(
        self, index: int, item: ProbePositionsRepositoryItem
    ) -> None:
        pass

    def handle_probe_changed(self, index: int, item: ProbeRepositoryItem) -> None:
        pass

    def handle_object_changed(self, index: int, item: ObjectRepositoryItem) -> None:
        pass

    def handle_losses_changed(self, index: int, losses: Sequence[LossValue]) -> None:
        if index == self._widget.currentIndex():
            self._status_controller.plot_losses(index)

    def handle_dataset_changed(self, index: int, item: ProductRepositoryItem) -> None:
        pass

    def handle_state_changed(self, index: int, item: ProductRepositoryItem) -> None:
        pass

    def handle_item_removed(self, index: int, item: ProductRepositoryItem) -> None:
        pass


class ComputeParameterViewController(ParameterViewController):
    def __init__(
        self,
        *,
        globus_supported: bool = True,
        genesis_supported: bool = True,
        tool_tip: str = '',
    ) -> None:
        self._local_button = QRadioButton('Local')
        self._globus_button = QRadioButton('Remote (Globus)')
        self._genesis_button = QRadioButton('Remote (Genesis)')
        self._button_group = QButtonGroup()
        self._widget = QWidget()

        if tool_tip:
            self._widget.setToolTip(tool_tip)

        self._button_group.addButton(self._local_button)
        self._button_group.addButton(self._globus_button)
        self._button_group.addButton(self._genesis_button)
        self._button_group.setExclusive(True)
        self._local_button.setChecked(True)
        self._globus_button.setVisible(globus_supported)
        self._genesis_button.setVisible(genesis_supported)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._local_button)
        layout.addWidget(self._globus_button)
        layout.addWidget(self._genesis_button)
        layout.addStretch()
        self._widget.setLayout(layout)

    def is_globus_button_checked(self) -> bool:
        return self._button_group.checkedButton() is self._globus_button

    def is_genesis_button_checked(self) -> bool:
        return self._button_group.checkedButton() is self._genesis_button

    def get_widget(self) -> QWidget:
        return self._widget

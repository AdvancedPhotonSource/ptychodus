from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
import logging

from PyQt5.QtCore import Qt, QModelIndex
from PyQt5.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QScrollArea,
    QStackedWidget,
    QWidget,
)

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.product import LossValue
from ptychodus.api.reconstructor import TrainableReconstructor

from ..model.globus import GlobusCore
from ..model.product import ProductRepository, ProductRepositoryItem, ProductRepositoryObserver
from ..model.processing import (
    ProcessingAPI,
    ProcessingAlgorithmParameter,
    ProcessingProgressMonitor,
)
from ..model.product.metadata import MetadataRepositoryItem
from ..model.product.object import ObjectRepositoryItem
from ..model.product.probe import ProbeRepositoryItem
from ..model.product.probe_positions import ProbePositionsRepositoryItem
from ..view.processing import ProcessingActionsView, ProcessingStatusView
from ..view.widgets import ExceptionDialog
from .data import FileDialogFactory
from .helpers import connect_triggered_signal
from .parametric import ComboBoxParameterViewController, ParameterViewController
from .product.list_model import ProductRepositoryListModel

logger = logging.getLogger(__name__)


class ReconstructorViewControllerFactory(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def create_view_controller(self, reconstructor_name: str) -> QWidget:
        pass


class ProcessingStatusController(Observer):
    def __init__(
        self,
        product_repository: ProductRepository,
        monitor: ProcessingProgressMonitor,
        view: ProcessingStatusView,
    ) -> None:
        super().__init__()
        self._product_repository = product_repository
        self._monitor = monitor
        self._view = view
        self._view.text_edit.setReadOnly(True)

        self._sync_model_to_view()
        monitor.add_observer(self)

    def plot_losses(self, product_index: int) -> None:
        # FIXME plot and progress can mismatch
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
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.grid(True)
        ax.plot(epoch, losses, '.-', label='Loss', linewidth=1.5)
        self._view.figure_canvas.draw()

    def _sync_model_to_view(self) -> None:
        # FIXME cancel processing; status messages to comments at end
        for text in self._monitor.messages():
            self._view.text_edit.appendPlainText(text)

        progress_goal = self._monitor.get_progress_goal()
        progress_bar = self._view.progress_bar

        if self._monitor.is_processing and progress_goal > 0:
            progress_bar.show()
            progress_bar.setRange(0, progress_goal)
            progress_bar.setValue(self._monitor.get_progress())
        else:
            progress_bar.hide()

    def _update(self, observable: Observable) -> None:
        if observable is self._monitor:
            self._sync_model_to_view()


class ProductParameterViewController(ParameterViewController, ProductRepositoryObserver):
    def __init__(
        self,
        repository: ProductRepository,
        status_controller: ProcessingStatusController,
        *,
        tool_tip: str = '',
    ) -> None:
        super().__init__()
        self._repository = repository
        self._status_controller = status_controller
        self._model = ProductRepositoryListModel(repository)
        self._widget = QComboBox()

        if tool_tip:
            self._widget.setToolTip(tool_tip)

        self._widget.setModel(self._model)
        self._widget.currentIndexChanged.connect(status_controller.plot_losses)

        repository.add_observer(self)

    def get_widget(self) -> QComboBox:
        return self._widget

    def handle_item_inserted(self, index: int, item: ProductRepositoryItem) -> None:
        parent = QModelIndex()
        self._model.beginInsertRows(parent, index, index)
        self._model.endInsertRows()

    def handle_metadata_changed(self, index: int, item: MetadataRepositoryItem) -> None:
        top_left = self._model.index(index, 0)
        bottom_right = self._model.index(index, 0)
        self._model.dataChanged.emit(top_left, bottom_right, [Qt.ItemDataRole.DisplayRole])

    def handle_probe_positions_changed(
        self, index: int, item: ProbePositionsRepositoryItem
    ) -> None:
        pass

    def handle_probe_changed(self, index: int, item: ProbeRepositoryItem) -> None:
        pass

    def handle_object_changed(self, index: int, item: ObjectRepositoryItem) -> None:
        pass

    def handle_losses_changed(self, index: int, losses: Sequence[LossValue]) -> None:
        current_index = self._widget.currentIndex()

        if index == current_index:
            self._status_controller.plot_losses(index)

    def handle_item_removed(self, index: int, item: ProductRepositoryItem) -> None:
        parent = QModelIndex()
        self._model.beginRemoveRows(parent, index, index)
        self._model.endRemoveRows()


class ComputeParameterViewController(ParameterViewController):
    def __init__(self, *, tool_tip: str = '') -> None:
        self._local_button = QRadioButton('Local')
        self._remote_button = QRadioButton('Remote')
        self._button_group = QButtonGroup()
        self._widget = QWidget()

        if tool_tip:
            self._widget.setToolTip(tool_tip)

        self._button_group.addButton(self._local_button, 0)
        self._button_group.addButton(self._remote_button)
        self._button_group.setExclusive(True)
        self._local_button.setChecked(True)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._local_button)
        layout.addWidget(self._remote_button)
        layout.addStretch()
        self._widget.setLayout(layout)

    def is_computing_local(self) -> bool:
        return self._button_group.checkedId() == 0

    def get_widget(self) -> QWidget:
        return self._widget


class ProcessingController(Observer):
    def __init__(
        self,
        algorithm_parameter: ProcessingAlgorithmParameter,
        processing_api: ProcessingAPI,
        product_repository: ProductRepository,
        globus: GlobusCore,
        view: QWidget,
        status_view: ProcessingStatusView,
        file_dialog_factory: FileDialogFactory,
        view_controller_factories: Iterable[ReconstructorViewControllerFactory],
    ) -> None:
        super().__init__()
        self._algorithm_parameter = algorithm_parameter
        self._processing_api = processing_api
        self._product_repository = product_repository
        self._globus = globus
        self._view = view
        self._status_view = status_view
        self._file_dialog_factory = file_dialog_factory
        self._view_controller_factories: dict[str, ReconstructorViewControllerFactory] = {
            vcf.name: vcf for vcf in view_controller_factories
        }

        self._stacked_widget = QStackedWidget()
        stacked_widget_layout = self._stacked_widget.layout()

        if stacked_widget_layout is not None:
            stacked_widget_layout.setContentsMargins(0, 0, 0, 0)

        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setWidget(self._stacked_widget)

        self._algorithm_view_controller = ComboBoxParameterViewController(
            algorithm_parameter, algorithm_parameter.available_reconstructors()
        )
        self._algorithm_view_controller.get_widget().currentIndexChanged.connect(
            self._stacked_widget.setCurrentIndex
        )
        self._status_controller = ProcessingStatusController(
            product_repository, processing_api.get_progress_monitor(), status_view
        )
        self._product_view_controller = ProductParameterViewController(
            product_repository, self._status_controller
        )
        self._compute_view_controller = ComputeParameterViewController()
        self._actions_view = ProcessingActionsView()

        layout = QFormLayout()
        layout.addRow('Algorithm:', self._algorithm_view_controller.get_widget())
        layout.addRow('Product:', self._product_view_controller.get_widget())

        if globus.is_supported:
            layout.addRow('Compute:', self._compute_view_controller.get_widget())

        layout.addRow('Action:', self._actions_view)
        layout.addRow(self._scroll_area)
        self._view.setLayout(layout)

        self._actions_view.reconstruct_button.clicked.connect(self._reconstruct)
        self._actions_view.train_button.clicked.connect(self._train)

        # TODO reconstruct split
        load_model_action = self._actions_view.actions_menu.addAction('Load Model...')
        connect_triggered_signal(load_model_action, self._load_model)
        export_training_data_action = self._actions_view.actions_menu.addAction(
            'Export Training Data...'
        )
        connect_triggered_signal(export_training_data_action, self._export_training_data)

        self._populate_stacked_widget()
        self._sync_model_to_view()
        algorithm_parameter.add_observer(self)

    def _populate_stacked_widget(self) -> None:
        for library, reconstructor in self._processing_api.available_reconstructors_parts():
            try:
                vcf = self._view_controller_factories[library]
            except KeyError:
                label = QLabel(f'{library} not found!')
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                widget: QWidget = label
            else:
                widget = vcf.create_view_controller(reconstructor)

            self._stacked_widget.addWidget(widget)

    def _reconstruct(self) -> None:
        input_product_index = self._product_view_controller.get_widget().currentIndex()

        if input_product_index < 0:
            return

        if self._compute_view_controller.is_computing_local():
            try:
                output_product_index = self._processing_api.reconstruct(input_product_index)
            except Exception as exc:
                logger.exception(exc)
                ExceptionDialog.show_exception('Reconstruct Local', exc)
            else:
                self._product_view_controller.get_widget().setCurrentIndex(output_product_index)
        else:
            try:
                self._globus.executor.reconstruct(input_product_index)
            except Exception as exc:
                logger.exception(exc)
                ExceptionDialog.show_exception('Reconstruct Remote', exc)

    def _train(self) -> None:
        product_index = self._product_view_controller.get_widget().currentIndex()

        if product_index < 0:
            return

        data_path = self._file_dialog_factory.get_existing_directory_path(
            self._view, 'Choose Training Data Directory'
        )

        if not data_path:
            return

        if self._compute_view_controller.is_computing_local():
            try:
                self._processing_api.train(product_index, data_path, data_path)
            except Exception as exc:
                logger.exception(exc)
                ExceptionDialog.show_exception('Train Local', exc)
        else:
            try:
                self._globus.executor.train(product_index)
            except Exception as exc:
                logger.exception(exc)
                ExceptionDialog.show_exception('Train Remote', exc)

    def _load_model(self) -> None:
        name_filter = self._processing_api.get_model_file_filter()
        file_path, name_filter = self._file_dialog_factory.get_open_file_path(
            self._view, 'Load Model', name_filters=[name_filter], selected_name_filter=name_filter
        )

        if not file_path:
            return

        try:
            self._processing_api.load_model_from_file(file_path)
        except Exception as exc:
            logger.exception(exc)
            ExceptionDialog.show_exception('Load Model', exc)

    def _export_training_data(self) -> None:
        input_product_index = self._product_view_controller.get_widget().currentIndex()

        if input_product_index < 0:
            return

        name_filter = self._processing_api.get_training_data_file_filter()
        file_path, _ = self._file_dialog_factory.get_save_file_path(
            self._view,
            'Export Training Data',
            name_filters=[name_filter],
            selected_name_filter=name_filter,
        )

        if not file_path:
            return

        try:
            self._processing_api.export_training_data(file_path, input_product_index)
        except Exception as exc:
            logger.exception(exc)
            ExceptionDialog.show_exception('Export Training Data', exc)

    def _sync_model_to_view(self) -> None:
        reconstructor = self._algorithm_parameter.get_current_reconstructor()
        is_trainable = isinstance(reconstructor, TrainableReconstructor)

        # FIXME hide TrainableReconstructor actions if not is_trainable

        if is_trainable:
            is_model_loaded = True  # FIXME
            self._actions_view.reconstruct_button.setText('Infer')
            self._actions_view.reconstruct_button.setEnabled(is_model_loaded)
        else:
            self._actions_view.reconstruct_button.setText('Reconstruct')
            self._actions_view.reconstruct_button.setEnabled(True)

        self._actions_view.train_button.setVisible(is_trainable)

    def _update(self, observable: Observable) -> None:
        if observable is self._algorithm_parameter:
            self._sync_model_to_view()

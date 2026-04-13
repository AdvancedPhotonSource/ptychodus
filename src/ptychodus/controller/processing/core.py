from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterable
import logging

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QFormLayout,
    QLabel,
    QScrollArea,
    QStackedWidget,
    QWidget,
)

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.reconstructor import TrainableReconstructor

from ...model.genesis import GenesisCore
from ...model.globus import GlobusCore
from ...model.processing import ProcessingAPI, ProcessingAlgorithmParameter
from ...model.product import ProductRepository
from ...view.processing import ProcessingActionsView, ProcessingStatusView
from ...view.widgets import ExceptionDialog
from ..data import FileDialogFactory
from ..helpers import connect_triggered_signal
from ..parametric import ComboBoxParameterViewController
from .parameters import (
    ProcessingStatusController,
    ProductParameterViewController,
    ComputeParameterViewController,
)

logger = logging.getLogger(__name__)


class ReconstructorViewControllerFactory(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def create_view_controller(self, reconstructor_name: str) -> QWidget:
        pass


class ProcessingController(Observer):
    def __init__(
        self,
        algorithm_parameter: ProcessingAlgorithmParameter,
        processing_api: ProcessingAPI,
        product_repository: ProductRepository,
        globus: GlobusCore,
        genesis: GenesisCore,
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
        self._genesis = genesis
        self._view = view
        self._status_view = status_view
        self._file_dialog_factory = file_dialog_factory

        self._stacked_widget = QStackedWidget()
        self._populate_stacked_widget(view_controller_factories)
        stacked_widget_layout = self._stacked_widget.layout()

        if stacked_widget_layout is not None:
            stacked_widget_layout.setContentsMargins(0, 0, 0, 0)

        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setWidget(self._stacked_widget)

        self._algorithm_view_controller = ComboBoxParameterViewController(
            algorithm_parameter, algorithm_parameter.available_reconstructors()
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

        self._reconstruct_split_action = self._actions_view.actions_menu.addAction(
            'Reconstruct Split'
        )
        self._reconstruct_split_action.setEnabled(False)  # TODO
        connect_triggered_signal(self._reconstruct_split_action, self._reconstruct_split)
        self._load_model_action = self._actions_view.actions_menu.addAction('Load Model...')
        connect_triggered_signal(self._load_model_action, self._load_model)
        self._export_training_data_action = self._actions_view.actions_menu.addAction(
            'Export Training Data...'
        )
        connect_triggered_signal(self._export_training_data_action, self._export_training_data)

        self._sync_model_to_view()
        algorithm_parameter.add_observer(self)

    def _populate_stacked_widget(
        self, view_controller_factories: Iterable[ReconstructorViewControllerFactory]
    ) -> None:
        vcf_map = {vcf.name.casefold(): vcf for vcf in view_controller_factories}

        for library, reconstructor in self._processing_api.available_reconstructors_parts():
            try:
                vcf = vcf_map[library]
            except KeyError:
                label = QLabel(f'{library} not found!')
                logger.warning(label.text())
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                widget: QWidget = label
            else:
                widget = vcf.create_view_controller(reconstructor)

            self._stacked_widget.addWidget(widget)

    def _reconstruct(self) -> None:
        input_product_index = self._product_view_controller.get_widget().currentIndex()

        if input_product_index < 0:
            return

        if self._compute_view_controller.is_globus_button_checked():
            try:
                self._globus.executor.reconstruct(input_product_index)
            except Exception as exc:
                logger.exception(exc)
                ExceptionDialog.show_exception('Reconstruct Remote (Globus)', exc)
        elif self._compute_view_controller.is_genesis_button_checked():
            try:
                self._genesis.executor.reconstruct(input_product_index)
            except Exception as exc:
                logger.exception(exc)
                ExceptionDialog.show_exception('Reconstruct Remote (Genesis)', exc)
        else:  # local
            try:
                output_product_index = self._processing_api.reconstruct(input_product_index)
            except Exception as exc:
                logger.exception(exc)
                ExceptionDialog.show_exception('Reconstruct Local', exc)
            else:
                self._product_view_controller.get_widget().setCurrentIndex(output_product_index)

    def _train(self) -> None:
        product_index = self._product_view_controller.get_widget().currentIndex()

        if product_index < 0:
            return

        data_path = self._file_dialog_factory.get_existing_directory_path(
            self._view, 'Choose Training Data Directory'
        )

        if not data_path:
            return

        if self._compute_view_controller.is_globus_button_checked():
            try:
                self._globus.executor.train(product_index)
            except Exception as exc:
                logger.exception(exc)
                ExceptionDialog.show_exception('Train Remote (Globus)', exc)
        elif self._compute_view_controller.is_genesis_button_checked():
            try:
                self._genesis.executor.train(product_index)
            except Exception as exc:
                logger.exception(exc)
                ExceptionDialog.show_exception('Train Remote (Genesis)', exc)
        else:  # local
            try:
                self._processing_api.train(product_index, data_path, data_path)
            except Exception as exc:
                logger.exception(exc)
                ExceptionDialog.show_exception('Train Local', exc)

    def _reconstruct_split(self) -> None:
        pass  # TODO

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
        self._stacked_widget.setCurrentIndex(
            self._algorithm_view_controller.get_widget().currentIndex()
        )
        reconstructor = self._algorithm_parameter.get_current_reconstructor()
        is_trainable = False

        if isinstance(reconstructor, TrainableReconstructor):
            is_trainable = True
            is_model_loaded = reconstructor.is_model_loaded()
            self._actions_view.reconstruct_button.setText('Infer')
            self._actions_view.reconstruct_button.setEnabled(is_model_loaded)
        else:
            self._actions_view.reconstruct_button.setText('Reconstruct')
            self._actions_view.reconstruct_button.setEnabled(True)

        self._actions_view.train_button.setVisible(is_trainable)
        self._load_model_action.setVisible(is_trainable)
        self._export_training_data_action.setVisible(is_trainable)

    def _update(self, observable: Observable) -> None:
        if observable is self._algorithm_parameter:
            self._sync_model_to_view()

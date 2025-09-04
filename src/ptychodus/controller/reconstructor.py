from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
import logging

from PyQt5.QtCore import Qt, QAbstractItemModel, QTimer
from PyQt5.QtWidgets import QActionGroup, QLabel, QWidget

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.product import LossValue

from ..model.product import (
    ProductRepository,
    ProductRepositoryItem,
    ProductRepositoryObserver,
)
from ..model.product.metadata import MetadataRepositoryItem
from ..model.product.object import ObjectRepositoryItem
from ..model.product.probe import ProbeRepositoryItem
from ..model.product.scan import ScanRepositoryItem
from ..model.reconstructor import ReconstructorPresenter
from ..view.reconstructor import ReconstructorView, ReconstructorPlotView
from ..view.widgets import ExceptionDialog
from .data import FileDialogFactory
from .helpers import connect_triggered_signal

logger = logging.getLogger(__name__)


class ReconstructorViewControllerFactory(ABC):
    @property
    @abstractmethod
    def backend_name(self) -> str:
        pass

    @abstractmethod
    def create_view_controller(self, reconstructor_name: str) -> QWidget:
        pass


class ReconstructorController(ProductRepositoryObserver, Observer):
    def __init__(
        self,
        presenter: ReconstructorPresenter,
        product_repository: ProductRepository,
        view: ReconstructorView,
        plot_view: ReconstructorPlotView,
        product_table_model: QAbstractItemModel,
        file_dialog_factory: FileDialogFactory,
        view_controller_factories: Iterable[ReconstructorViewControllerFactory],
    ) -> None:
        super().__init__()
        self._presenter = presenter
        self._product_repository = product_repository
        self._view = view
        self._plot_view = plot_view
        self._file_dialog_factory = file_dialog_factory
        self._view_controller_factories: dict[str, ReconstructorViewControllerFactory] = {
            vcf.backend_name: vcf for vcf in view_controller_factories
        }

        for name in presenter.reconstructors():
            self._add_reconstructor(name)

        view.parameters_view.algorithm_combo_box.textActivated.connect(presenter.set_reconstructor)
        view.parameters_view.algorithm_combo_box.currentIndexChanged.connect(
            view.stacked_widget.setCurrentIndex
        )

        view.parameters_view.product_combo_box.textActivated.connect(self._redraw_plot)
        view.parameters_view.product_combo_box.setModel(product_table_model)

        self._progress_timer = QTimer()
        self._progress_timer.timeout.connect(self._update_progress)
        self._progress_timer.start(5 * 1000)  # TODO customize (in milliseconds)

        view.progress_dialog.setModal(True)
        view.progress_dialog.setWindowModality(Qt.WindowModality.ApplicationModal)
        view.progress_dialog.setWindowFlags(
            Qt.WindowType.Window | Qt.WindowType.WindowTitleHint | Qt.WindowType.CustomizeWindowHint
        )
        view.progress_dialog.text_edit.setReadOnly(True)

        open_model_action = view.parameters_view.reconstructor_menu.addAction('Open Model...')
        connect_triggered_signal(open_model_action, self._open_model)
        save_model_action = view.parameters_view.reconstructor_menu.addAction('Save Model...')
        connect_triggered_signal(save_model_action, self._save_model)

        self._model_action_group = QActionGroup(view.parameters_view.reconstructor_menu)
        self._model_action_group.setExclusive(False)
        self._model_action_group.addAction(open_model_action)
        self._model_action_group.addAction(save_model_action)
        self._model_action_group.addAction(view.parameters_view.reconstructor_menu.addSeparator())

        reconstruct_transformed_action = view.parameters_view.reconstructor_menu.addAction(
            'Reconstruct Transformed Points'
        )
        connect_triggered_signal(reconstruct_transformed_action, self._reconstruct_transformed)
        reconstruct_split_action = view.parameters_view.reconstructor_menu.addAction(
            'Reconstruct Odd/Even Split'
        )
        connect_triggered_signal(reconstruct_split_action, self._reconstruct_split)
        reconstruct_action = view.parameters_view.reconstructor_menu.addAction('Reconstruct')
        connect_triggered_signal(reconstruct_action, self._reconstruct)

        export_training_data_action = view.parameters_view.trainer_menu.addAction(
            'Export Training Data...'
        )
        connect_triggered_signal(export_training_data_action, self._export_training_data)
        train_action = view.parameters_view.trainer_menu.addAction('Train')
        connect_triggered_signal(train_action, self._train)

        presenter.add_observer(self)
        product_repository.add_observer(self)
        self._sync_model_to_view()

    def _update_progress(self) -> None:
        is_reconstructing = self._presenter.is_reconstructing

        for button in self._view.progress_dialog.button_box.buttons():
            button.setEnabled(not is_reconstructing)

        for text in self._presenter.flush_log():
            self._view.progress_dialog.text_edit.appendPlainText(text)

        self._presenter.process_results(block=False)

    def _add_reconstructor(self, name: str) -> None:
        backend_name, reconstructor_name = name.split('/')  # TODO REDO
        self._view.parameters_view.algorithm_combo_box.addItem(
            name, self._view.parameters_view.algorithm_combo_box.count()
        )

        if backend_name in self._view_controller_factories:
            view_controller_factory = self._view_controller_factories[backend_name]
            widget = view_controller_factory.create_view_controller(reconstructor_name)
        else:
            widget = QLabel(f'{backend_name} not found!')
            widget.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._view.stacked_widget.addWidget(widget)

    def _show_progress_dialog(self) -> None:
        self._view.progress_dialog.show()
        self._update_progress()

    def _reconstruct(self) -> None:
        input_product_index = self._view.parameters_view.product_combo_box.currentIndex()

        if input_product_index < 0:
            return

        try:
            output_product_index = self._presenter.reconstruct(input_product_index)
        except Exception as exc:
            logger.exception(exc)
            ExceptionDialog.show_exception('Reconstructor', exc)
        else:
            self._view.parameters_view.product_combo_box.setCurrentIndex(output_product_index)
            self._show_progress_dialog()

    def _reconstruct_split(self) -> None:
        input_product_index = self._view.parameters_view.product_combo_box.currentIndex()

        if input_product_index < 0:
            return

        try:
            self._presenter.reconstruct_split(input_product_index)
        except Exception as exc:
            logger.exception(exc)
            ExceptionDialog.show_exception('Split Reconstructor', exc)

        self._show_progress_dialog()

    def _reconstruct_transformed(self) -> None:
        input_product_index = self._view.parameters_view.product_combo_box.currentIndex()

        if input_product_index < 0:
            return

        try:
            self._presenter.reconstruct_transformed(input_product_index)
        except Exception as exc:
            logger.exception(exc)
            ExceptionDialog.show_exception('Split Reconstructor', exc)

        self._show_progress_dialog()

    def _open_model(self) -> None:
        name_filter = self._presenter.get_model_file_filter()
        file_path, name_filter = self._file_dialog_factory.get_open_file_path(
            self._view, 'Open Model', name_filters=[name_filter], selected_name_filter=name_filter
        )

        if file_path:
            try:
                self._presenter.open_model(file_path)
            except Exception as exc:
                logger.exception(exc)
                ExceptionDialog.show_exception('Model Reader', exc)

    def _save_model(self) -> None:
        name_filter = self._presenter.get_model_file_filter()
        file_path, _ = self._file_dialog_factory.get_save_file_path(
            self._view, 'Save Model', name_filters=[name_filter], selected_name_filter=name_filter
        )

        if file_path:
            try:
                self._presenter.save_model(file_path)
            except Exception as exc:
                logger.exception(exc)
                ExceptionDialog.show_exception('Model Writer', exc)

    def _export_training_data(self) -> None:
        input_product_index = self._view.parameters_view.product_combo_box.currentIndex()

        if input_product_index < 0:
            return

        name_filter = self._presenter.get_training_data_file_filter()
        file_path, _ = self._file_dialog_factory.get_save_file_path(
            self._view,
            'Export Training Data',
            name_filters=[name_filter],
            selected_name_filter=name_filter,
        )

        if file_path:
            try:
                self._presenter.export_training_data(file_path, input_product_index)
            except Exception as exc:
                logger.exception(exc)
                ExceptionDialog.show_exception('Training Data Writer', exc)

    def _train(self) -> None:
        data_path = self._file_dialog_factory.get_existing_directory_path(
            self._view,
            'Choose Training Data Directory',
            initial_directory=self._presenter.get_training_data_path(),
        )

        if data_path:
            try:
                self._presenter.train(data_path)
            except Exception as exc:
                logger.exception(exc)
                ExceptionDialog.show_exception('Trainer', exc)

    def _redraw_plot(self) -> None:
        product_index = self._view.parameters_view.product_combo_box.currentIndex()

        if product_index < 0:
            self._plot_view.axes.clear()
            return

        try:
            item = self._product_repository[product_index]
        except IndexError as exc:
            logger.exception(exc)
            return

        epoch = [loss.epoch for loss in item.get_losses()]
        losses = [loss.value for loss in item.get_losses()]

        ax = self._plot_view.axes
        ax.clear()
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.grid(True)
        ax.plot(epoch, losses, '.-', label='Loss', linewidth=1.5)
        self._plot_view.figure_canvas.draw()

    def _sync_model_to_view(self) -> None:
        self._view.parameters_view.algorithm_combo_box.setCurrentText(
            self._presenter.get_reconstructor()
        )

        is_trainable = self._presenter.is_trainable
        self._model_action_group.setVisible(is_trainable)
        self._view.parameters_view.trainer_button.setVisible(is_trainable)

        self._redraw_plot()

    def handle_item_inserted(self, index: int, item: ProductRepositoryItem) -> None:
        pass

    def handle_metadata_changed(self, index: int, item: MetadataRepositoryItem) -> None:
        pass

    def handle_scan_changed(self, index: int, item: ScanRepositoryItem) -> None:
        pass

    def handle_probe_changed(self, index: int, item: ProbeRepositoryItem) -> None:
        pass

    def handle_object_changed(self, index: int, item: ObjectRepositoryItem) -> None:
        pass

    def handle_losses_changed(self, index: int, losses: Sequence[LossValue]) -> None:
        current_index = self._view.parameters_view.product_combo_box.currentIndex()

        if index == current_index:
            self._redraw_plot()

    def handle_item_removed(self, index: int, item: ProductRepositoryItem) -> None:
        pass

    def _update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._sync_model_to_view()

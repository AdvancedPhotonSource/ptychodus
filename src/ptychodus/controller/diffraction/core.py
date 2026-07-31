import logging


from pathlib import Path

from PyQt5.QtCore import QModelIndex
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.parametric import PathParameter, StringParameter

from ...model.analysis import DiffractionSimulator, DiffractionSimulatorSettings
from ...model.diffraction import (
    AssembledDiffractionDataset,
    DetectorSettings,
    DiffractionAPI,
    DiffractionDatasetObserver,
    DiffractionDatasetRepository,
    DiffractionDatasetRepositoryObserver,
    DiffractionSettings,
    DiffractionTaskMonitor,
    PatternSizer,
)
from ...model.metadata import MetadataPresenter
from ...model.product import ProductRepository
from ...view.diffraction import DetectorView, DiffractionStatusView, PatternsView
from ...view.widgets import ExceptionDialog, ProgressBarItemDelegate
from ..data import FileDialogFactory
from ..helpers import connect_triggered_signal
from ..image import ImageController
from ..parametric import (
    CheckBoxParameterViewController,
    LengthWidgetParameterViewController,
    ParameterViewController,
    SpinBoxParameterViewController,
)
from ..product.list_model import ProductRepositoryListModel
from .dataset import DatasetTreeModel
from .dataset_layout import DatasetLayoutViewController
from .wizard import OpenDatasetWizardController

logger = logging.getLogger(__name__)


class BadPixelsViewController(ParameterViewController):
    """Edits the bad-pixels settings; the mask is applied when a dataset is (re)loaded."""

    # FIXME move to wizard because this is now per-dataset.

    def __init__(
        self,
        bad_pixels_file_path: PathParameter,
        bad_pixels_file_type: StringParameter,
        repository: DiffractionDatasetRepository,
        diffraction_api: DiffractionAPI,
        file_dialog_factory: FileDialogFactory,
    ) -> None:
        super().__init__()
        self._bad_pixels_file_path = bad_pixels_file_path
        self._bad_pixels_file_type = bad_pixels_file_type
        self._repository = repository
        self._diffraction_api = diffraction_api
        self._file_dialog_factory = file_dialog_factory
        self._dataset_index = -1

        self._line_edit = QLineEdit()
        self._line_edit.setReadOnly(True)
        self._browse_button = QPushButton('Browse...')
        self._browse_button.clicked.connect(self._choose_bad_pixels_file)
        self._clear_button = QPushButton('Clear')
        self._clear_button.clicked.connect(self._clear_bad_pixels_setting)
        self._hint_label = QLabel('Reload dataset to apply.')
        self._widget = QWidget()

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self._line_edit)
        row.addWidget(self._browse_button)
        row.addWidget(self._clear_button)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(row)
        layout.addWidget(self._hint_label)
        self._widget.setLayout(layout)

        self._sync_model_to_view()

    def set_dataset_index(self, index: int) -> None:
        self._dataset_index = index
        self._sync_model_to_view()

    def _current_dataset_index(self) -> int | None:
        return self._dataset_index if self._dataset_index >= 0 else None

    def _choose_bad_pixels_file(self) -> None:
        file_reader_chooser = self._diffraction_api.get_bad_pixels_file_reader_chooser()
        current_plugin = file_reader_chooser.get_current_plugin()
        file_path, name_filter = self._file_dialog_factory.get_open_file_path(
            self._widget,
            'Open Bad Pixels File',
            name_filters=[plugin.display_name for plugin in file_reader_chooser],
            selected_name_filter=current_plugin.simple_name,
        )

        if file_path:
            self._bad_pixels_file_path.set_value(file_path)
            if name_filter:
                self._bad_pixels_file_type.set_value(name_filter)
            self._sync_model_to_view()

    def _clear_bad_pixels_setting(self) -> None:
        self._bad_pixels_file_path.set_value(Path())
        self._sync_model_to_view()

    def get_widget(self) -> QWidget:
        return self._widget

    def _sync_model_to_view(self) -> None:
        dataset_index = self._current_dataset_index()
        if dataset_index is None:
            self._line_edit.setText('0')
            return
        bad_pixels = self._repository[dataset_index].get_bad_pixels()
        self._line_edit.setText(str(int(bad_pixels.sum())))


class DetectorController:
    def __init__(
        self,
        settings: DetectorSettings,
        repository: DiffractionDatasetRepository,
        diffraction_api: DiffractionAPI,
        view: DetectorView,
        file_dialog_factory: FileDialogFactory,
    ) -> None:
        self._width_px_view_controller = SpinBoxParameterViewController(settings.width_px)
        self._height_px_view_controller = SpinBoxParameterViewController(settings.height_px)
        self._pixel_width_view_controller = LengthWidgetParameterViewController(
            settings.pixel_width_m
        )
        self._pixel_height_view_controller = LengthWidgetParameterViewController(
            settings.pixel_height_m
        )
        self._bad_pixels_view_controller = BadPixelsViewController(
            settings.bad_pixels_file_path,
            settings.bad_pixels_file_type,
            repository,
            diffraction_api,
            file_dialog_factory,
        )

        layout = QFormLayout()
        layout.addRow('Detector Width [px]:', self._width_px_view_controller.get_widget())
        layout.addRow('Detector Height [px]:', self._height_px_view_controller.get_widget())
        layout.addRow('Pixel Width:', self._pixel_width_view_controller.get_widget())
        layout.addRow('Pixel Height:', self._pixel_height_view_controller.get_widget())
        layout.addRow('Bad Pixels:', self._bad_pixels_view_controller.get_widget())
        view.setLayout(layout)

    def set_dataset_index(self, index: int) -> None:
        self._bad_pixels_view_controller.set_dataset_index(index)


class DiffractionStatusController(Observer):
    def __init__(
        self,
        monitor: DiffractionTaskMonitor,
        view: DiffractionStatusView,
    ) -> None:
        super().__init__()
        self._monitor = monitor
        self._view = view

        view.stop_button.clicked.connect(monitor.stop_processing)

        self._sync_model_to_view()
        monitor.add_observer(self)

    def _sync_model_to_view(self) -> None:
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


class DiffractionController(DiffractionDatasetRepositoryObserver):
    def __init__(
        self,
        detector_settings: DetectorSettings,
        diffraction_settings: DiffractionSettings,
        pattern_sizer: PatternSizer,
        diffraction_api: DiffractionAPI,
        repository: DiffractionDatasetRepository,
        task_monitor: DiffractionTaskMonitor,
        metadata_presenter: MetadataPresenter,
        product_repository: ProductRepository,
        diffraction_simulator: DiffractionSimulator,
        diffraction_simulator_settings: DiffractionSimulatorSettings,
        view: PatternsView,
        status_view: DiffractionStatusView,
        image_controller: ImageController,
        file_dialog_factory: FileDialogFactory,
    ) -> None:
        super().__init__()
        self._pattern_sizer = pattern_sizer
        self._diffraction_api = diffraction_api
        self._repository = repository
        self._current_dataset_index = -1
        self._product_list_model = ProductRepositoryListModel(product_repository)
        self._diffraction_simulator = diffraction_simulator
        self._view = view
        self._image_controller = image_controller
        self._file_dialog_factory = file_dialog_factory
        self._per_dataset_observers: dict[int, _PerDatasetObserver] = {}
        self._detector_controller = DetectorController(
            detector_settings,
            repository,
            diffraction_api,
            view.detector_view,
            file_dialog_factory,
        )
        self._wizard_controller = OpenDatasetWizardController(
            diffraction_settings,
            detector_settings,
            diffraction_api,
            repository,
            metadata_presenter,
            file_dialog_factory,
        )
        self._status_controller = DiffractionStatusController(task_monitor, status_view)
        self._tree_model = DatasetTreeModel()

        view.tree_view.setModel(self._tree_model)
        view.tree_view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        header = view.tree_view.header()
        header.setSectionResizeMode(header.ResizeMode.ResizeToContents)
        counts_item_delegate = ProgressBarItemDelegate(view.tree_view)
        view.tree_view.setItemDelegateForColumn(1, counts_item_delegate)
        selection_model = view.tree_view.selectionModel()

        if selection_model is None:
            raise ValueError('selection_model is None!')
        else:
            selection_model.currentChanged.connect(self._on_tree_selection_changed)

        self._image_controller.clear_array()

        open_dataset_action = view.button_box.load_menu.addAction('Open File...')
        connect_triggered_signal(open_dataset_action, self._wizard_controller.open_dataset)

        view.button_box.save_button.clicked.connect(self._save_dataset)
        remove_selected_action = view.button_box.close_menu.addAction('Remove Selected Dataset')
        connect_triggered_signal(remove_selected_action, self._remove_selected_dataset)
        close_all_action = view.button_box.close_menu.addAction('Close All Datasets')
        connect_triggered_signal(close_all_action, self._close_all_datasets)

        dataset_layout_action = view.button_box.analyze_menu.addAction('Dataset Layout...')
        connect_triggered_signal(dataset_layout_action, self._show_dataset_layout)
        simulate_action = view.button_box.analyze_menu.addAction('Simulate Diffraction...')
        connect_triggered_signal(simulate_action, self._choose_product_for_simulation)

        view.simulate_dialog.product_combo_box.setModel(self._product_list_model)
        self._poisson_view_controller = CheckBoxParameterViewController(
            diffraction_simulator_settings.add_poisson_noise,
            'Add Poisson Noise',
        )
        view.simulate_dialog.form_layout.insertRow(1, self._poisson_view_controller.get_widget())
        view.simulate_dialog.finished.connect(self._simulate_diffraction)

        repository.add_observer(self)

        # Populate the tree for datasets already present at construction time.
        for index in range(len(repository)):
            self._on_dataset_inserted(index, repository[index])

        self._update_info_text()

    def _current_dataset(self) -> AssembledDiffractionDataset | None:
        index = self._current_dataset_index
        return self._repository[index] if 0 <= index < len(self._repository) else None

    def _set_current_dataset_index(self, index: int) -> None:
        self._current_dataset_index = index
        self._detector_controller.set_dataset_index(index)

    def _on_tree_selection_changed(self, current: QModelIndex, previous: QModelIndex) -> None:
        # Update the image preview based on the selected tree node.
        if current.isValid():
            node = current.internalPointer()
            data = node.get_data()
            if data is not None:
                pixel_geometry = self._pattern_sizer.get_processed_pixel_geometry()
                self._image_controller.set_array(data, pixel_geometry)
            else:
                self._image_controller.clear_array()
        else:
            self._image_controller.clear_array()

        # And track the containing dataset as the panel's current dataset.
        dataset_row = self._tree_model.dataset_row_for_index(current)
        self._set_current_dataset_index(dataset_row if dataset_row is not None else -1)

    def _save_dataset(self) -> None:
        dataset_index = self._current_dataset_index
        if dataset_index < 0:
            return

        file_writer_chooser = self._diffraction_api.get_file_writer_chooser()
        file_path, name_filter = self._file_dialog_factory.get_save_file_path(
            self._view,
            'Save Diffraction File',
            name_filters=[plugin.display_name for plugin in file_writer_chooser],
            selected_name_filter=file_writer_chooser.get_current_plugin().display_name,
        )

        if file_path:
            try:
                self._diffraction_api.save_patterns(
                    file_path, name_filter, dataset_index=dataset_index
                )
            except Exception as exc:
                logger.exception(exc)
                ExceptionDialog.show_exception('File Writer', exc)

    def _show_dataset_layout(self) -> None:
        current = self._current_dataset()
        if current is not None:
            DatasetLayoutViewController.show_dialog(current, self._view)

    def _choose_product_for_simulation(self) -> None:
        self._product_list_model.beginResetModel()
        self._product_list_model.endResetModel()
        self._view.simulate_dialog.open()

    def _simulate_diffraction(self, result: int) -> None:
        if result != QDialog.DialogCode.Accepted:
            return
        item_index = self._view.simulate_dialog.product_combo_box.currentIndex()
        if item_index < 0:
            logger.warning('Cannot simulate diffraction: no product selected.')
            return
        self._diffraction_simulator.simulate(item_index)

    def _remove_selected_dataset(self) -> None:
        dataset_index = self._current_dataset_index
        if dataset_index < 0:
            return

        button = QMessageBox.question(
            self._view,
            'Confirm Remove',
            'Remove the selected diffraction dataset from memory?',
        )

        if button == QMessageBox.StandardButton.Yes:
            self._diffraction_api.close_patterns(dataset_index)
            self._image_controller.clear_array()

    def _close_all_datasets(self) -> None:
        if len(self._repository) == 0:
            return

        button = QMessageBox.question(
            self._view,
            'Confirm Close All',
            'Free every loaded diffraction dataset from memory?',
        )

        if button == QMessageBox.StandardButton.Yes:
            self._diffraction_api.close_all_patterns()
            self._image_controller.clear_array()

    def _update_info_text(self) -> None:
        self._view.info_label.setText(self._repository.get_info_text())

    def _select_dataset_row(self, dataset_row: int) -> None:
        selection_model = self._view.tree_view.selectionModel()
        if selection_model is None:
            return
        model_index = self._tree_model.index(dataset_row, 0)
        if model_index.isValid():
            selection_model.setCurrentIndex(
                model_index,
                selection_model.SelectionFlag.ClearAndSelect | selection_model.SelectionFlag.Rows,
            )

    def _on_dataset_inserted(self, index: int, dataset: AssembledDiffractionDataset) -> None:
        self._tree_model.insert_dataset(index, dataset)

        observer = _PerDatasetObserver(self, dataset)
        dataset.add_observer(observer)
        self._per_dataset_observers[id(dataset)] = observer

        # Populate any arrays already present on the dataset.
        for array_index in range(len(dataset)):
            self._tree_model.insert_array(index, array_index, dataset[array_index])

        self._update_info_text()

    def handle_dataset_inserted(self, index: int, dataset: AssembledDiffractionDataset) -> None:
        self._on_dataset_inserted(index, dataset)
        # Make a freshly loaded dataset the panel's current dataset.
        self._select_dataset_row(index)

    def handle_dataset_removed(self, index: int, dataset: AssembledDiffractionDataset) -> None:
        observer = self._per_dataset_observers.pop(id(dataset), None)
        if observer is not None:
            dataset.remove_observer(observer)
        self._tree_model.remove_dataset(index)
        self._update_info_text()

        # Re-sync the panel's current dataset with the (possibly changed) tree selection.
        selection_model = self._view.tree_view.selectionModel()
        current = selection_model.currentIndex() if selection_model is not None else QModelIndex()
        dataset_row = self._tree_model.dataset_row_for_index(current)
        self._set_current_dataset_index(dataset_row if dataset_row is not None else -1)

    def _dataset_row(self, dataset: AssembledDiffractionDataset) -> int | None:
        try:
            return list(self._repository).index(dataset)
        except ValueError:
            return None

    def _handle_array_inserted_for_dataset(
        self, dataset: AssembledDiffractionDataset, array_row: int
    ) -> None:
        dataset_row = self._dataset_row(dataset)
        if dataset_row is None:
            return
        self._tree_model.insert_array(dataset_row, array_row, dataset[array_row])
        self._update_info_text()

    def _handle_array_changed_for_dataset(
        self, dataset: AssembledDiffractionDataset, array_row: int
    ) -> None:
        dataset_row = self._dataset_row(dataset)
        if dataset_row is None:
            return
        self._tree_model.refresh_array(dataset_row, array_row)

    def _handle_dataset_reloaded_for_dataset(self, dataset: AssembledDiffractionDataset) -> None:
        dataset_row = self._dataset_row(dataset)
        if dataset_row is None:
            return
        self._tree_model.refresh_dataset(dataset_row)


class _PerDatasetObserver(DiffractionDatasetObserver):
    """Per-dataset observer wrapper that captures the dataset ref at registration."""

    def __init__(
        self, controller: DiffractionController, dataset: AssembledDiffractionDataset
    ) -> None:
        super().__init__()
        self._controller = controller
        self._dataset = dataset

    def handle_array_inserted(self, index: int) -> None:
        self._controller._handle_array_inserted_for_dataset(self._dataset, index)

    def handle_array_changed(self, index: int) -> None:
        self._controller._handle_array_changed_for_dataset(self._dataset, index)

    def handle_dataset_reloaded(self) -> None:
        self._controller._handle_dataset_reloaded_for_dataset(self._dataset)

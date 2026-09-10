import logging

from PyQt5.QtCore import QModelIndex
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QMessageBox,
    QStatusBar,
)

from ...model.analysis import DiffractionSimulator, DiffractionSimulatorSettings
from ...model.diffraction import (
    AssembledDiffractionDataset,
    DetectorSettings,
    DiffractionAPI,
    DiffractionDatasetObserver,
    DiffractionDatasetRepository,
    DiffractionDatasetRepositoryObserver,
    DiffractionSettings,
    DiffractionSummaryService,
    DiffractionTaskMonitor,
)
from .detector_extent import DetectorExtentSource
from ...model.product import ProductRepository, ProductSettings
from ...model.visualization import VisualizationEngine
from ...view.diffraction import DiffractionView
from ...view.widgets import ExceptionDialog, ProgressBarItemDelegate, TaskStatusView
from ..data import FileDialogFactory
from ..helpers import connect_triggered_signal
from ..image import ImageController
from ..parameters import CheckBoxParameterViewController
from ..task_status import TaskStatusController
from ..product.core import ProductRepositoryComboProxyModel, ProductRepositoryTableModel
from .dataset import DiffractionTreeModel
from .dataset_editor import DatasetEditorViewController
from .wizard import OpenDatasetWizardController

logger = logging.getLogger(__name__)


class DiffractionController(DiffractionDatasetRepositoryObserver):
    def __init__(
        self,
        detector_settings: DetectorSettings,
        diffraction_settings: DiffractionSettings,
        product_settings: ProductSettings,
        diffraction_api: DiffractionAPI,
        repository: DiffractionDatasetRepository,
        task_monitor: DiffractionTaskMonitor,
        product_repository: ProductRepository,
        product_table_model: ProductRepositoryTableModel,
        diffraction_simulator: DiffractionSimulator,
        diffraction_simulator_settings: DiffractionSimulatorSettings,
        view: DiffractionView,
        status_view: TaskStatusView,
        image_controller: ImageController,
        summary_service: DiffractionSummaryService,
        summary_visualization_engine: VisualizationEngine,
        status_bar: QStatusBar,
        file_dialog_factory: FileDialogFactory,
    ) -> None:
        super().__init__()
        self._diffraction_api = diffraction_api
        self._repository = repository
        self._detector_extent_source = DetectorExtentSource()
        self._current_dataset_index = -1
        self._product_combo_model = ProductRepositoryComboProxyModel(
            product_table_model, product_repository
        )
        self._diffraction_simulator = diffraction_simulator
        self._view = view
        self._image_controller = image_controller
        self._file_dialog_factory = file_dialog_factory
        self._per_dataset_observers: dict[int, _PerDatasetObserver] = {}
        self._wizard_controller = OpenDatasetWizardController(
            diffraction_settings,
            detector_settings,
            product_settings,
            self._detector_extent_source,
            diffraction_api,
            repository,
            summary_service,
            summary_visualization_engine,
            status_bar,
            file_dialog_factory,
        )
        self._status_controller = TaskStatusController(task_monitor, status_view)
        self._tree_model = DiffractionTreeModel(repository)

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
            selection_model.currentChanged.connect(self._update_enabled_buttons)

        self._image_controller.clear_array()

        open_dataset_action = view.button_box.insert_menu.addAction('Open File...')
        connect_triggered_signal(open_dataset_action, self._wizard_controller.open_dataset)
        simulate_action = view.button_box.insert_menu.addAction('Simulate...')
        connect_triggered_signal(simulate_action, self._choose_product_for_simulation)

        save_file_action = view.button_box.save_menu.addAction('Save File...')
        connect_triggered_signal(save_file_action, self._save_dataset)

        view.button_box.edit_button.clicked.connect(self._edit_current_dataset)
        view.button_box.remove_button.clicked.connect(self._remove_selected_dataset)

        view.simulate_dialog.product_combo_box.setModel(self._product_combo_model)
        self._poisson_view_controller = CheckBoxParameterViewController(
            diffraction_simulator_settings.add_poisson_noise,
            'Add Poisson Noise',
        )
        view.simulate_dialog.form_layout.insertRow(1, self._poisson_view_controller.get_widget())
        view.simulate_dialog.finished.connect(self._simulate_diffraction)

        self._refresh_enabled_buttons()
        repository.add_observer(self)

        # The tree model builds its own rows for pre-existing datasets; the panel only
        # needs its own observer on each.
        for dataset in repository:
            self._attach_panel_observer(dataset)

        self._update_info_text()

    def _current_dataset(self) -> AssembledDiffractionDataset | None:
        index = self._current_dataset_index
        return self._repository[index] if 0 <= index < len(self._repository) else None

    def _set_current_dataset_index(self, index: int) -> None:
        self._current_dataset_index = index

    def _on_tree_selection_changed(self, current: QModelIndex, previous: QModelIndex) -> None:
        # Update the image preview based on the selected tree node.
        if current.isValid():
            node = current.internalPointer()
            data = node.get_data()
            if data is not None:
                dataset_row = self._tree_model.dataset_row_for_index(current)
                dataset = (
                    self._repository[dataset_row]
                    if dataset_row is not None and 0 <= dataset_row < len(self._repository)
                    else None
                )
                if dataset is not None:
                    pixel_geometry = dataset.get_processed_pixel_geometry()
                    self._image_controller.set_array(data, pixel_geometry)
                else:
                    self._image_controller.clear_array()
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

        file_path, name_filter = self._file_dialog_factory.get_save_file_path(
            self._view,
            'Save Diffraction File',
            name_filters=[nf for nf in self._diffraction_api.get_save_file_filters()],
            selected_name_filter=self._diffraction_api.get_save_file_filter(),
        )

        if file_path:
            try:
                self._diffraction_api.save_patterns(
                    file_path, name_filter, dataset_index=dataset_index
                )
            except Exception as exc:
                logger.exception(exc)
                ExceptionDialog.show_exception('File Writer', exc)

    def _edit_current_dataset(self) -> None:
        current = self._current_dataset()
        if current is not None:
            DatasetEditorViewController.edit_dataset(current, self._view)

    def _choose_product_for_simulation(self) -> None:
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

    def _update_info_text(self) -> None:
        self._view.info_label.setText(self._repository.get_info_text())

    def _refresh_enabled_buttons(self) -> None:
        """Re-run the button gating against whatever the tree currently has selected.

        Used for programmatic refreshes; the currentChanged signal calls
        _update_enabled_buttons directly with the new index.
        """
        self._update_enabled_buttons(self._view.tree_view.currentIndex(), QModelIndex())

    def _update_enabled_buttons(self, current: QModelIndex, previous: QModelIndex) -> None:
        dataset_row = self._tree_model.dataset_row_for_index(current)
        dataset = (
            self._repository[dataset_row]
            if dataset_row is not None and 0 <= dataset_row < len(self._repository)
            else None
        )
        ready = dataset is not None and not dataset.is_load_in_progress()

        self._view.button_box.save_button.setEnabled(ready)
        self._view.button_box.edit_button.setEnabled(ready)
        # Remove is always safe: it drops the row whether pending, failed, or ready.
        self._view.button_box.remove_button.setEnabled(dataset is not None)

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

    def _attach_panel_observer(self, dataset: AssembledDiffractionDataset) -> None:
        observer = _PerDatasetObserver(self, dataset)
        dataset.add_observer(observer)
        self._per_dataset_observers[id(dataset)] = observer

    def handle_dataset_inserted(self, index: int, dataset: AssembledDiffractionDataset) -> None:
        # Row construction belongs to the tree model, which observes the repository too.
        self._attach_panel_observer(dataset)
        self._update_info_text()
        # Publish the newly-loaded detector extent to the wizard's crop/bin bounds.
        self._detector_extent_source.set_extent(dataset.get_metadata().detector_extent)

        # Select the new dataset only when nothing is selected, matching the Products
        # and Fluorescence panels; a load must not yank the selection out from under
        # someone inspecting another dataset.
        if not self._view.tree_view.currentIndex().isValid():
            self._select_dataset_row(index)

    def handle_dataset_removed(self, index: int, dataset: AssembledDiffractionDataset) -> None:
        observer = self._per_dataset_observers.pop(id(dataset), None)

        if observer is not None:
            dataset.remove_observer(observer)

        self._update_info_text()

        # Pick a neighbour rather than leaving the panel with nothing selected, as the
        # Products and Fluorescence panels do.
        selection_model = self._view.tree_view.selectionModel()
        current = selection_model.currentIndex() if selection_model is not None else QModelIndex()

        if not current.isValid():
            row_count = self._tree_model.rowCount()

            if row_count > 0:
                self._select_dataset_row(min(index, row_count - 1))
                current = (
                    selection_model.currentIndex() if selection_model is not None else QModelIndex()
                )

        dataset_row = self._tree_model.dataset_row_for_index(current)
        self._set_current_dataset_index(dataset_row if dataset_row is not None else -1)
        self._refresh_enabled_buttons()

    def handle_metadata_changed(self, index: int, dataset: AssembledDiffractionDataset) -> None:
        self._update_info_text()

    def handle_state_changed(self, index: int, dataset: AssembledDiffractionDataset) -> None:
        if index == self._current_dataset_index:
            self._refresh_enabled_buttons()

    def _dataset_row(self, dataset: AssembledDiffractionDataset) -> int | None:
        try:
            return list(self._repository).index(dataset)
        except ValueError:
            return None

    def _handle_array_inserted_for_dataset(
        self, dataset: AssembledDiffractionDataset, array_row: int
    ) -> None:
        # The tree model inserts the row; the panel only updates its own summary and
        # buttons, since a landing array can flip the current dataset out of the
        # loading state and re-enable Save / Edit.
        self._update_info_text()

        if self._dataset_row(dataset) == self._current_dataset_index:
            self._refresh_enabled_buttons()

    def _handle_dataset_reloaded_for_dataset(self, dataset: AssembledDiffractionDataset) -> None:
        # Refresh the wizard's crop/bin bounds when a dataset is reloaded (e.g. via a
        # streaming context or a programmatic re-open with a different file).
        self._detector_extent_source.set_extent(dataset.get_metadata().detector_extent)


class _PerDatasetObserver(DiffractionDatasetObserver):
    """Panel-level per-dataset observer.

    Row updates are the tree model's job (it registers its own observer); this one
    carries only the two reactions the panel owns — the summary label plus button
    gating when an array lands, and the wizard's detector extent on reload.
    """

    def __init__(
        self, controller: DiffractionController, dataset: AssembledDiffractionDataset
    ) -> None:
        super().__init__()
        self._controller = controller
        self._dataset = dataset

    def handle_array_inserted(self, index: int) -> None:
        self._controller._handle_array_inserted_for_dataset(self._dataset, index)

    def handle_array_changed(self, index: int) -> None:
        pass

    def handle_dataset_reloaded(self) -> None:
        self._controller._handle_dataset_reloaded_for_dataset(self._dataset)

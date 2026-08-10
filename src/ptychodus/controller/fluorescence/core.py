from __future__ import annotations
import logging

from PyQt5.QtCore import QModelIndex
from PyQt5.QtWidgets import QInputDialog

from ptychodus.api.observer import Observable, Observer

from ...model.fluorescence import (
    FluorescenceAPI,
    FluorescenceItemState,
    FluorescenceRepository,
    FluorescenceRepositoryItem,
    FluorescenceRepositoryObserver,
)
from ...model.product import ProductRepository
from ...view.fluorescence import FluorescenceView
from ...view.widgets import ExceptionDialog
from ..data import FileDialogFactory
from ..image import ImageController
from .enhance_dialog import FluorescenceEnhanceDialogController
from .repository_tree_model import FluorescenceRepositoryTreeModel, Variant

logger = logging.getLogger(__name__)


class FluorescenceController(FluorescenceRepositoryObserver, Observer):
    """Top-level controller for the promoted Fluorescence subview.

    Owns the left-pane dataset browser, the right-pane element viewer, and
    the modal enhance dialog controller.

    The dataset tree expands to element leaves (see
    :class:`FluorescenceRepositoryTreeModel`); selecting an item renders a
    summary of all element maps, selecting an element renders that map. A
    Measured/Enhanced radio pair below the tree selects which variant drives
    Counts + rendering.
    """

    def __init__(
        self,
        repository: FluorescenceRepository,
        api: FluorescenceAPI,
        product_repository: ProductRepository,
        view: FluorescenceView,
        image_controller: ImageController,
        enhance_dialog_controller: FluorescenceEnhanceDialogController,
        file_dialog_factory: FileDialogFactory,
    ) -> None:
        super().__init__()
        self._repository = repository
        self._api = api
        self._product_repository = product_repository
        self._view = view
        self._image_controller = image_controller
        self._enhance_dialog_controller = enhance_dialog_controller
        self._file_dialog_factory = file_dialog_factory
        self._task_monitor = api.get_task_monitor()

        self._tree_model = FluorescenceRepositoryTreeModel(repository)
        view.tree_view.setModel(self._tree_model)
        header = view.tree_view.header()
        header.setSectionResizeMode(header.ResizeMode.ResizeToContents)

        selection_model = view.tree_view.selectionModel()
        if selection_model is None:
            raise ValueError('tree_view selection model is None!')
        selection_model.currentChanged.connect(self._on_tree_selection_changed)

        view.measured_radio_button.toggled.connect(self._on_variant_toggled)
        view.enhanced_radio_button.toggled.connect(self._on_variant_toggled)

        view.button_box.load_button.clicked.connect(self._load)
        view.button_box.enhance_button.clicked.connect(self._enhance)
        view.button_box.save_button.clicked.connect(self._save)
        view.button_box.remove_button.clicked.connect(self._remove)

        repository.add_observer(self)
        self._task_monitor.add_observer(self)

        self._sync_variant_toggle_enabled()
        self._sync_buttons()

    # ------------------------------------------------------------------
    # Selection & rendering
    # ------------------------------------------------------------------

    def _current_variant(self) -> Variant:
        return 'enhanced' if self._view.enhanced_radio_button.isChecked() else 'measured'

    def _current_top_level_row(self) -> int:
        return self._tree_model.item_row_for_index(self._view.tree_view.currentIndex())

    def _current_item(self) -> FluorescenceRepositoryItem | None:
        row = self._current_top_level_row()
        if row < 0:
            return None
        try:
            return self._repository[row]
        except IndexError:
            return None

    def _on_tree_selection_changed(self, current: QModelIndex, previous: QModelIndex) -> None:
        self._sync_variant_toggle_enabled()
        self._render_current()
        self._sync_buttons()

    def _on_variant_toggled(self, checked: bool) -> None:
        # Signals fire twice per toggle (button loses check + button gains
        # check). Handle only the gain to avoid duplicate re-renders.
        if not checked:
            return
        self._tree_model.set_variant(self._current_variant())
        self._render_current()

    def _render_current(self) -> None:
        current = self._view.tree_view.currentIndex()
        if not current.isValid():
            self._image_controller.clear_array()
            return
        node = current.internalPointer()
        item = self._current_item()
        if node is None or item is None:
            self._image_controller.clear_array()
            return

        variant = self._current_variant()
        array = node.get_data(item, variant)
        if array is None:
            self._image_controller.clear_array()
            return

        pixel_geometry = item.get_product().get_geometry().get_object_plane_pixel_geometry()
        self._image_controller.set_array(array, pixel_geometry)

    # ------------------------------------------------------------------
    # Variant toggle enable-state
    # ------------------------------------------------------------------

    def _sync_variant_toggle_enabled(self) -> None:
        item = self._current_item()
        has_enhanced = item is not None and item.get_enhanced() is not None
        self._view.enhanced_radio_button.setEnabled(has_enhanced)
        # If the current item can't be shown enhanced, fall back to measured
        # (and let toggled -> re-render handle the redraw).
        if not has_enhanced and self._view.enhanced_radio_button.isChecked():
            self._view.measured_radio_button.setChecked(True)

    # ------------------------------------------------------------------
    # Button actions
    # ------------------------------------------------------------------

    def _choose_product(self, title: str) -> tuple[bool, int]:
        """Prompt the user to pick the target product for a new fluorescence dataset.

        Returns (accepted, product_index). product_index is -1 when the user
        cancels or when the product repository is empty (message shown).
        Mirrors the pattern in ProductController._choose_dataset.
        """
        names = [
            self._product_repository[i].get_name() for i in range(len(self._product_repository))
        ]

        if not names:
            ExceptionDialog.show_exception(
                title,
                ValueError('No products loaded — create or open a product first.'),
            )
            return False, -1

        label, accepted = QInputDialog.getItem(
            self._view,
            title,
            'Target Product:',
            names,
            0,
            False,
        )

        if not accepted:
            return False, -1

        return True, names.index(label)

    def _load(self) -> None:
        title = 'Open Measured Fluorescence Dataset'
        # Pick product first so cancelling doesn't waste a file selection.
        accepted, product_index = self._choose_product(title)
        if not accepted:
            return

        file_path, name_filter = self._file_dialog_factory.get_open_file_path(
            self._view,
            title,
            name_filters=list(self._api.get_open_file_filters()),
            selected_name_filter=self._api.get_open_file_filter(),
        )
        if not file_path:
            return
        try:
            self._api.open_measured_dataset(file_path, product_index, file_type=name_filter)
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.show_exception(title, err)

    def _enhance(self) -> None:
        row = self._current_top_level_row()
        if row < 0:
            return
        self._enhance_dialog_controller.launch(row)

    def _save(self) -> None:
        row = self._current_top_level_row()
        if row < 0:
            return
        try:
            item = self._repository[row]
        except IndexError:
            return
        if item.get_enhanced() is None:
            return
        title = 'Save Enhanced Fluorescence Dataset'
        file_path, name_filter = self._file_dialog_factory.get_save_file_path(
            self._view,
            title,
            name_filters=list(self._api.get_save_file_filters()),
            selected_name_filter=self._api.get_save_file_filter(),
        )
        if not file_path:
            return
        try:
            self._api.save_enhanced_dataset(row, file_path, file_type=name_filter)
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.show_exception(title, err)

    def _remove(self) -> None:
        row = self._current_top_level_row()
        if row < 0:
            return
        self._api.remove_item(row)

    def _sync_buttons(self) -> None:
        item = self._current_item()
        has_item = item is not None
        has_enhanced = item is not None and item.get_enhanced() is not None
        can_enhance = (
            item is not None
            and item.get_state() is FluorescenceItemState.READY
            and not self._task_monitor.is_processing
        )

        self._view.button_box.load_button.setEnabled(True)
        self._view.button_box.enhance_button.setEnabled(can_enhance)
        self._view.button_box.save_button.setEnabled(has_enhanced)
        self._view.button_box.remove_button.setEnabled(has_item)

    # ------------------------------------------------------------------
    # Observers
    # ------------------------------------------------------------------

    def handle_item_inserted(self, index: int, item: FluorescenceRepositoryItem) -> None:
        if not self._view.tree_view.currentIndex().isValid():
            self._view.tree_view.setCurrentIndex(self._tree_model.index(index, 0))
        self._sync_variant_toggle_enabled()
        self._sync_buttons()

    def handle_item_removed(self, index: int, item: FluorescenceRepositoryItem) -> None:
        if not self._view.tree_view.currentIndex().isValid():
            row_count = self._tree_model.rowCount()
            if row_count > 0:
                target = min(index, row_count - 1)
                self._view.tree_view.setCurrentIndex(self._tree_model.index(target, 0))
            else:
                self._render_current()
        self._sync_variant_toggle_enabled()
        self._sync_buttons()

    def handle_metadata_changed(self, index: int, item: FluorescenceRepositoryItem) -> None:
        if index == self._current_top_level_row():
            self._sync_buttons()

    def handle_enhanced_changed(self, index: int, item: FluorescenceRepositoryItem) -> None:
        if index == self._current_top_level_row():
            self._sync_variant_toggle_enabled()
            self._render_current()
            self._sync_buttons()

    def handle_state_changed(self, index: int, item: FluorescenceRepositoryItem) -> None:
        if index == self._current_top_level_row():
            if item.get_state() is FluorescenceItemState.FAILED:
                logger.warning(f'Enhancement failed for item "{item.get_label()}"')
            elif item.get_state() is FluorescenceItemState.ORPHANED:
                logger.info(f'Fluorescence item "{item.get_label()}" orphaned by product removal')
            self._sync_buttons()

    def _update(self, observable: Observable) -> None:
        if observable is self._task_monitor:
            self._sync_buttons()

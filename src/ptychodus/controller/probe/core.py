from __future__ import annotations
import logging

from PyQt5.QtCore import QModelIndex, QStringListModel
from PyQt5.QtWidgets import QAbstractItemView, QDialog

from ptychodus.api.observer import SequenceObserver

from ...model.analysis import (
    IlluminationMapper,
    ProbePropagator,
    STXMSimulator,
)
from ...model.fluorescence import FluorescenceEnhancer
from ...model.product import ProbeAPI, ProbeRepository
from ...model.product.probe import ProbeRepositoryItem
from ...model.visualization import VisualizationEngine
from ...view.repository import RepositoryTreeView
from ...view.widgets import (
    ComboBoxItemDelegate,
    ExceptionDialog,
    ProgressBarItemDelegate,
)
from ..data import FileDialogFactory
from ..helpers import connect_triggered_signal, create_brush_for_editable_cell
from ..image import ImageController
from .editor_factory import ProbeEditorViewControllerFactory
from .fluorescence import FluorescenceViewController
from .illumination import IlluminationViewController
from .propagator import ProbePropagationViewController
from .stxm import STXMViewController
from .tree_model import ProbeTreeModel

logger = logging.getLogger(__name__)


class ProbeController(SequenceObserver[ProbeRepositoryItem]):
    def __init__(
        self,
        repository: ProbeRepository,
        api: ProbeAPI,
        image_controller: ImageController,
        propagator: ProbePropagator,
        propagator_visualization_engine: VisualizationEngine,
        stxm_simulator: STXMSimulator,
        stxm_visualization_engine: VisualizationEngine,
        illumination_mapper: IlluminationMapper,
        illumination_visualization_engine: VisualizationEngine,
        fluorescence_enhancer: FluorescenceEnhancer,
        fluorescence_visualization_engine: VisualizationEngine,
        view: RepositoryTreeView,
        file_dialog_factory: FileDialogFactory,
        *,
        is_developer_mode_enabled: bool,
    ) -> None:
        super().__init__()
        self._repository = repository
        self._api = api
        self._image_controller = image_controller
        self._view = view
        self._file_dialog_factory = file_dialog_factory
        self._tree_model = ProbeTreeModel(repository, api, create_brush_for_editable_cell(view))
        self._editor_factory = ProbeEditorViewControllerFactory()

        self._propagation_view_controller = ProbePropagationViewController(
            propagator, propagator_visualization_engine, file_dialog_factory
        )
        self._stxm_view_controller = STXMViewController(
            stxm_simulator, stxm_visualization_engine, file_dialog_factory
        )
        self._illumination_view_controller = IlluminationViewController(
            illumination_mapper,
            illumination_visualization_engine,
            file_dialog_factory,
            is_developer_mode_enabled=is_developer_mode_enabled,
        )
        self._fluorescence_view_controller = FluorescenceViewController(
            fluorescence_enhancer, fluorescence_visualization_engine, file_dialog_factory
        )

        # TODO figure out good fix when saving NPY file without suffix (numpy adds suffix)
        repository.add_observer(self)

        builder_list_model = QStringListModel()
        builder_list_model.setStringList([name for name in api.builder_names()])
        builder_item_delegate = ComboBoxItemDelegate(builder_list_model, view.tree_view)

        view.tree_view.setModel(self._tree_model)
        view.tree_view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        power_item_delegate = ProgressBarItemDelegate(view.tree_view)
        view.tree_view.setItemDelegateForColumn(1, power_item_delegate)
        view.tree_view.setItemDelegateForColumn(2, builder_item_delegate)
        selection_model = view.tree_view.selectionModel()

        if selection_model is None:
            raise ValueError('selection_model is None!')
        else:
            selection_model.currentChanged.connect(self._update_view)

        self._update_view(QModelIndex(), QModelIndex())

        load_from_file_action = view.button_box.load_menu.addAction('Open File...')
        connect_triggered_signal(load_from_file_action, self._load_current_probe_from_file)

        copy_action = view.button_box.load_menu.addAction('Copy...')
        connect_triggered_signal(copy_action, self._copy_to_current_probe)

        save_to_file_action = view.button_box.save_menu.addAction('Save File...')
        connect_triggered_signal(save_to_file_action, self._save_current_probe_to_file)

        sync_to_settings_action = view.button_box.save_menu.addAction('Sync To Settings')
        connect_triggered_signal(sync_to_settings_action, self._sync_current_probe_to_settings)

        view.copier_dialog.setWindowTitle('Copy Probe')
        view.copier_dialog.source_combo_box.setModel(self._tree_model)
        view.copier_dialog.destination_combo_box.setModel(self._tree_model)
        view.copier_dialog.finished.connect(self._finish_copying_probe)

        view.button_box.edit_button.clicked.connect(self._edit_current_probe)

        propagate_action = view.button_box.analyze_menu.addAction('Propagate...')
        connect_triggered_signal(propagate_action, self._propagate_probe)

        stxm_action = view.button_box.analyze_menu.addAction('Simulate STXM...')
        connect_triggered_signal(stxm_action, self._simulate_stxm)

        illumination_action = view.button_box.analyze_menu.addAction('Map Illumination...')
        connect_triggered_signal(illumination_action, self._map_illumination)

        fluorescence_action = view.button_box.analyze_menu.addAction('Enhance Fluorescence...')
        connect_triggered_signal(fluorescence_action, self._enhance_fluorescence)

    def _get_current_item_index(self) -> int:
        model_index = self._view.tree_view.currentIndex()

        if model_index.isValid():
            parent = model_index.parent()

            while parent.isValid():
                model_index = parent
                parent = model_index.parent()

            return model_index.row()

        logger.warning('No current index!')
        return -1

    def _load_current_probe_from_file(self) -> None:
        item_index = self._get_current_item_index()

        if item_index < 0:
            return

        file_path, name_filter = self._file_dialog_factory.get_open_file_path(
            self._view,
            'Open Probe',
            name_filters=[nf for nf in self._api.get_open_file_filters()],
            selected_name_filter=self._api.get_open_file_filter(),
        )

        if file_path:
            try:
                self._api.open_probe(item_index, file_path, file_type=name_filter)
            except Exception as err:
                logger.exception(err)
                ExceptionDialog.show_exception('File Reader', err)

    def _copy_to_current_probe(self) -> None:
        item_index = self._get_current_item_index()

        if item_index >= 0:
            self._view.copier_dialog.destination_combo_box.setCurrentIndex(item_index)
            self._view.copier_dialog.open()

    def _finish_copying_probe(self, result: int) -> None:
        if result == QDialog.DialogCode.Accepted:
            source_index = self._view.copier_dialog.source_combo_box.currentIndex()
            destination_index = self._view.copier_dialog.destination_combo_box.currentIndex()
            self._api.copy_probe(source_index, destination_index)

    def _edit_current_probe(self) -> None:
        item_index = self._get_current_item_index()

        if item_index < 0:
            return

        item_name = self._repository.get_name(item_index)
        item = self._repository[item_index]
        dialog = self._editor_factory.create_editor_dialog(item_name, item, self._view)
        dialog.open()

    def _save_current_probe_to_file(self) -> None:
        item_index = self._get_current_item_index()

        if item_index < 0:
            return

        file_path, name_filter = self._file_dialog_factory.get_save_file_path(
            self._view,
            'Save Probe',
            name_filters=[nf for nf in self._api.get_save_file_filters()],
            selected_name_filter=self._api.get_save_file_filter(),
        )

        if file_path:
            try:
                self._api.save_probe(item_index, file_path, name_filter)
            except Exception as err:
                logger.exception(err)
                ExceptionDialog.show_exception('File Writer', err)

    def _sync_current_probe_to_settings(self) -> None:
        item_index = self._get_current_item_index()

        if item_index < 0:
            logger.warning('No current item!')
        else:
            item = self._repository[item_index]
            item.sync_to_settings()

    def _propagate_probe(self) -> None:
        item_index = self._get_current_item_index()

        if item_index < 0:
            logger.warning('No current item!')
        else:
            self._propagation_view_controller.launch(item_index)

    def _simulate_stxm(self) -> None:
        item_index = self._get_current_item_index()

        if item_index < 0:
            logger.warning('No current item!')
        else:
            self._stxm_view_controller.simulate(item_index)

    def _map_illumination(self) -> None:
        item_index = self._get_current_item_index()

        if item_index < 0:
            logger.warning('No current item!')
        else:
            self._illumination_view_controller.map(item_index)

    def _enhance_fluorescence(self) -> None:
        item_index = self._get_current_item_index()

        if item_index < 0:
            logger.warning('No current item!')
        else:
            self._fluorescence_view_controller.launch(item_index)

    def _update_view(self, current: QModelIndex, previous: QModelIndex) -> None:
        enabled = current.isValid()
        self._view.button_box.load_button.setEnabled(enabled)
        self._view.button_box.save_button.setEnabled(enabled)
        self._view.button_box.edit_button.setEnabled(enabled)
        self._view.button_box.analyze_button.setEnabled(enabled)

        item_index = self._get_current_item_index()

        if item_index < 0:
            self._image_controller.clear_array()
        else:
            try:
                item = self._repository[item_index]
            except IndexError:
                logger.warning('Unable to access item for visualization!')
            else:
                probe = item.get_probes().get_probe_no_opr()  # TODO OPR
                array = (
                    probe.get_incoherent_mode(current.row())
                    if current.parent().isValid()
                    else probe.get_incoherent_modes_flattened()
                )
                self._image_controller.set_array(array, probe.get_pixel_geometry())

    def handle_item_inserted(self, index: int, item: ProbeRepositoryItem) -> None:
        self._tree_model.insert_item(index, item)

    def handle_item_changed(self, index: int, item: ProbeRepositoryItem) -> None:
        self._tree_model.update_item(index, item)

        if index == self._get_current_item_index():
            current_index = self._view.tree_view.currentIndex()
            self._update_view(current_index, current_index)

    def handle_item_removed(self, index: int, item: ProbeRepositoryItem) -> None:
        self._tree_model.remove_item(index, item)

from __future__ import annotations
import logging

from PyQt5.QtCore import QModelIndex, QStringListModel
from PyQt5.QtWidgets import QAbstractItemView, QDialog

from ptychodus.api.observer import SequenceObserver

from ...model.analysis import FourierRingCorrelator, XMCDAnalyzer
from ...model.product import ObjectAPI, ObjectRepository
from ...model.product.object import ObjectRepositoryItem
from ...model.visualization import VisualizationEngine
from ...view.repository import RepositoryTreeView
from ...view.widgets import ComboBoxItemDelegate, ExceptionDialog
from ..data import FileDialogFactory
from ..image import ImageController
from .editor_factory import ObjectEditorViewControllerFactory
from .frc import FourierRingCorrelationViewController
from .tree_model import ObjectTreeModel
from .xmcd import XMCDViewController

logger = logging.getLogger(__name__)


class ObjectController(SequenceObserver[ObjectRepositoryItem]):
    def __init__(
        self,
        repository: ObjectRepository,
        api: ObjectAPI,
        image_controller: ImageController,
        correlator: FourierRingCorrelator,
        xmcd_analyzer: XMCDAnalyzer,
        xmcd_visualization_engine: VisualizationEngine,
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
        self._tree_model = ObjectTreeModel(repository, api)
        self._editor_factory = ObjectEditorViewControllerFactory()

        self._frc_view_controller = FourierRingCorrelationViewController(
            correlator, self._tree_model
        )
        self._xmcd_view_controller = XMCDViewController(
            xmcd_analyzer, xmcd_visualization_engine, file_dialog_factory, self._tree_model
        )

        # TODO figure out good fix when saving NPY file without suffix (numpy adds suffix)
        repository.add_observer(self)

        builder_list_model = QStringListModel()
        builder_list_model.setStringList([name for name in api.builder_names()])
        builder_item_delegate = ComboBoxItemDelegate(builder_list_model, view.tree_view)

        view.tree_view.setModel(self._tree_model)
        view.tree_view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        view.tree_view.setItemDelegateForColumn(2, builder_item_delegate)
        view.tree_view.selectionModel().currentChanged.connect(self._update_view)
        self._update_view(QModelIndex(), QModelIndex())

        load_from_file_action = view.button_box.load_menu.addAction('Open File...')
        load_from_file_action.triggered.connect(self._load_current_object_from_file)

        copy_action = view.button_box.load_menu.addAction('Copy...')
        copy_action.triggered.connect(self._copy_to_current_object)

        save_to_file_action = view.button_box.save_menu.addAction('Save File...')
        save_to_file_action.triggered.connect(self._save_current_object_to_file)

        sync_to_settings_action = view.button_box.save_menu.addAction('Sync To Settings')
        sync_to_settings_action.triggered.connect(self._sync_current_object_to_settings)

        view.copier_dialog.setWindowTitle('Copy Object')
        view.copier_dialog.source_combo_box.setModel(self._tree_model)
        view.copier_dialog.destination_combo_box.setModel(self._tree_model)
        view.copier_dialog.finished.connect(self._finish_copying_object)

        view.button_box.edit_button.clicked.connect(self._edit_current_object)

        frc_action = view.button_box.analyze_menu.addAction('Fourier Ring Correlation...')
        frc_action.triggered.connect(self._analyze_frc)

        xmcd_action = view.button_box.analyze_menu.addAction('XMCD...')
        xmcd_action.triggered.connect(self._analyze_xmcd)

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

    def _load_current_object_from_file(self) -> None:
        item_index = self._get_current_item_index()

        if item_index < 0:
            return

        file_path, name_filter = self._file_dialog_factory.get_open_file_path(
            self._view,
            'Open Object',
            name_filters=[nf for nf in self._api.get_open_file_filters()],
            selected_name_filter=self._api.get_open_file_filter(),
        )

        if file_path:
            try:
                self._api.open_object(item_index, file_path, file_type=name_filter)
            except Exception as err:
                logger.exception(err)
                ExceptionDialog.show_exception('File Reader', err)

    def _copy_to_current_object(self) -> None:
        item_index = self._get_current_item_index()

        if item_index >= 0:
            self._view.copier_dialog.destination_combo_box.setCurrentIndex(item_index)
            self._view.copier_dialog.open()

    def _finish_copying_object(self, result: int) -> None:
        if result == QDialog.DialogCode.Accepted:
            source_index = self._view.copier_dialog.source_combo_box.currentIndex()
            destination_index = self._view.copier_dialog.destination_combo_box.currentIndex()
            self._api.copy_object(source_index, destination_index)

    def _edit_current_object(self) -> None:
        item_index = self._get_current_item_index()

        if item_index < 0:
            return

        item_name = self._repository.get_name(item_index)
        item = self._repository[item_index]
        dialog = self._editor_factory.create_editor_dialog(item_name, item, self._view)
        dialog.open()

    def _save_current_object_to_file(self) -> None:
        item_index = self._get_current_item_index()

        if item_index < 0:
            return

        file_path, name_filter = self._file_dialog_factory.get_save_file_path(
            self._view,
            'Save Object',
            name_filters=[nameFilter for nameFilter in self._api.get_save_file_filters()],
            selected_name_filter=self._api.get_save_file_filter(),
        )

        if file_path:
            try:
                self._api.save_object(item_index, file_path, name_filter)
            except Exception as err:
                logger.exception(err)
                ExceptionDialog.show_exception('File Writer', err)

    def _sync_current_object_to_settings(self) -> None:
        item_index = self._get_current_item_index()

        if item_index < 0:
            logger.warning('No current item!')
        else:
            item = self._repository[item_index]
            item.sync_to_settings()

    def _analyze_frc(self) -> None:
        item_index = self._get_current_item_index()

        if item_index < 0:
            logger.warning('No current item!')
        else:
            self._frc_view_controller.analyze(item_index, item_index)

    def _analyze_xmcd(self) -> None:
        item_index = self._get_current_item_index()

        if item_index < 0:
            logger.warning('No current item!')
        else:
            self._xmcd_view_controller.analyze(item_index, item_index)

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
                object_ = item.get_object()
                array = (
                    object_.get_layer(current.row())
                    if current.parent().isValid()
                    else object_.get_layers_flattened()
                )
                self._image_controller.set_array(array, object_.get_pixel_geometry())

    def handle_item_inserted(self, index: int, item: ObjectRepositoryItem) -> None:
        self._tree_model.insert_item(index, item)

    def handle_item_changed(self, index: int, item: ObjectRepositoryItem) -> None:
        self._tree_model.update_item(index, item)

        if index == self._get_current_item_index():
            current_index = self._view.tree_view.currentIndex()
            self._update_view(current_index, current_index)

    def handle_item_removed(self, index: int, item: ObjectRepositoryItem) -> None:
        self._tree_model.remove_item(index, item)

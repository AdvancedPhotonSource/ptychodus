import logging


from PyQt5.QtCore import QModelIndex
from PyQt5.QtWidgets import QAbstractItemView, QFormLayout, QMessageBox

from ...model.metadata import MetadataPresenter
from ...model.patterns import (
    AssembledDiffractionDataset,
    DetectorSettings,
    DiffractionDatasetObserver,
    PatternSettings,
    PatternSizer,
    PatternsAPI,
)
from ...view.patterns import DetectorView, PatternsView
from ...view.widgets import ExceptionDialog, ProgressBarItemDelegate
from ..data import FileDialogFactory
from ..image import ImageController
from ..parametric import LengthWidgetParameterViewController, SpinBoxParameterViewController
from .dataset import DatasetTreeModel
from .info import PatternsInfoViewController
from .wizard import OpenDatasetWizardController

logger = logging.getLogger(__name__)


class DetectorController:
    def __init__(self, settings: DetectorSettings, view: DetectorView) -> None:
        self._width_px_view_controller = SpinBoxParameterViewController(settings.width_px)
        self._height_px_view_controller = SpinBoxParameterViewController(settings.height_px)
        self._pixel_width_view_controller = LengthWidgetParameterViewController(
            settings.pixel_width_m
        )
        self._pixel_height_view_controller = LengthWidgetParameterViewController(
            settings.pixel_height_m
        )
        self._bit_depth_view_controller = SpinBoxParameterViewController(settings.bit_depth)

        layout = QFormLayout()
        layout.addRow('Detector Width [px]:', self._width_px_view_controller.get_widget())
        layout.addRow('Detector Height [px]:', self._height_px_view_controller.get_widget())
        layout.addRow('Pixel Width:', self._pixel_width_view_controller.get_widget())
        layout.addRow('Pixel Height:', self._pixel_height_view_controller.get_widget())
        layout.addRow('Bit Depth:', self._bit_depth_view_controller.get_widget())
        view.setLayout(layout)


class PatternsController(DiffractionDatasetObserver):
    def __init__(
        self,
        detector_settings: DetectorSettings,
        pattern_settings: PatternSettings,
        pattern_sizer: PatternSizer,
        patterns_api: PatternsAPI,
        dataset: AssembledDiffractionDataset,
        metadata_presenter: MetadataPresenter,
        view: PatternsView,
        image_controller: ImageController,
        file_dialog_factory: FileDialogFactory,
    ) -> None:
        super().__init__()
        self._pattern_sizer = pattern_sizer
        self._patterns_api = patterns_api
        self._dataset = dataset
        self._view = view
        self._image_controller = image_controller
        self._file_dialog_factory = file_dialog_factory
        self._detector_controller = DetectorController(detector_settings, view.detector_view)
        self._wizard_controller = OpenDatasetWizardController(
            pattern_settings,
            pattern_sizer,
            patterns_api,
            metadata_presenter,
            file_dialog_factory,
        )
        self._tree_model = DatasetTreeModel()

        view.tree_view.setModel(self._tree_model)
        view.tree_view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        counts_item_delegate = ProgressBarItemDelegate(view.tree_view)
        view.tree_view.setItemDelegateForColumn(1, counts_item_delegate)
        view.tree_view.selectionModel().currentChanged.connect(self._update_view)
        self._update_view(QModelIndex(), QModelIndex())

        view.button_box.open_button.clicked.connect(self._wizard_controller.open_dataset)
        view.button_box.save_button.clicked.connect(self._save_dataset)
        view.button_box.info_button.clicked.connect(self._open_patterns_info)
        view.button_box.close_button.clicked.connect(self._close_dataset)
        dataset.add_observer(self)

        self._sync_model_to_view()

    def _update_view(self, current: QModelIndex, previous: QModelIndex) -> None:
        if current.isValid():
            node = current.internalPointer()
            data = node.get_data()
            pixel_geometry = self._pattern_sizer.get_processed_pixel_geometry()
            self._image_controller.set_array(data, pixel_geometry)
        else:
            self._image_controller.clear_array()

    def _save_dataset(self) -> None:
        file_writer_chooser = self._patterns_api.get_file_writer_chooser()
        file_path, name_filter = self._file_dialog_factory.get_save_file_path(
            self._view,
            'Save Diffraction File',
            name_filters=[plugin.display_name for plugin in file_writer_chooser],
            selected_name_filter=file_writer_chooser.get_current_plugin().display_name,
        )

        if file_path:
            try:
                self._patterns_api.save_patterns(file_path, name_filter)
            except Exception as err:
                logger.exception(err)
                ExceptionDialog.show_exception('File Writer', err)

    def _open_patterns_info(self) -> None:
        PatternsInfoViewController.show_info(self._dataset, self._view)

    def _close_dataset(self) -> None:
        button = QMessageBox.question(
            self._view,
            'Confirm Close',
            'This will free the diffraction data from memory. Do you want to continue?',
        )

        if button == QMessageBox.StandardButton.Yes:
            self._patterns_api.close_patterns()

    def _sync_model_to_view(self) -> None:
        self._tree_model.clear()

        for index, array in enumerate(self._dataset):
            self._tree_model.insert_array(index, array)  # type: ignore

        info_text = self._dataset.get_info_text()
        self._view.info_label.setText(info_text)

    def handle_array_inserted(self, index: int) -> None:
        self._tree_model.insert_array(index, self._dataset[index])

    def handle_array_changed(self, index: int) -> None:
        self._tree_model.refresh_array(index)

    def handle_dataset_reloaded(self) -> None:
        self._sync_model_to_view()

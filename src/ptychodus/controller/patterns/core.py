import logging


from PyQt5.QtCore import QModelIndex
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QWidget,
)

from ptychodus.api.parametric import PathParameter, StringParameter

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
from ..parametric import (
    LengthWidgetParameterViewController,
    ParameterViewController,
    SpinBoxParameterViewController,
)
from .dataset import DatasetTreeModel
from .dataset_layout import DatasetLayoutViewController
from .wizard import OpenDatasetWizardController

logger = logging.getLogger(__name__)


class BadPixelsViewController(ParameterViewController):
    def __init__(
        self,
        bad_pixels_file_path: PathParameter,
        bad_pixels_file_type: StringParameter,
        patterns_api: PatternsAPI,
        file_dialog_factory: FileDialogFactory,
    ) -> None:
        self._bad_pixels_file_path = bad_pixels_file_path
        self._bad_pixels_file_type = bad_pixels_file_type
        self._patterns_api = patterns_api
        self._file_dialog_factory = file_dialog_factory

        self._line_edit = QLineEdit()
        self._line_edit.setReadOnly(True)
        self._browse_button = QPushButton('Browse...')
        self._browse_button.clicked.connect(self._open_bad_pixels)
        self._clear_button = QPushButton('Clear')
        self._clear_button.clicked.connect(self._patterns_api.clear_bad_pixels)
        self._widget = QWidget()

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._line_edit)
        layout.addWidget(self._browse_button)
        layout.addWidget(self._clear_button)
        self._widget.setLayout(layout)

        self.set_num_bad_pixels(0)

    def _open_bad_pixels(self) -> None:
        file_reader_chooser = self._patterns_api.get_bad_pixels_file_reader_chooser()
        current_plugin = file_reader_chooser.get_current_plugin()
        file_path, name_filter = self._file_dialog_factory.get_open_file_path(
            self._widget,
            'Open Bad Pixels File',
            name_filters=[plugin.display_name for plugin in file_reader_chooser],
            selected_name_filter=current_plugin.simple_name,
        )

        if file_path:
            try:
                self._patterns_api.open_bad_pixels(file_path, file_type=name_filter)
            except Exception as exc:
                logger.exception(exc)
                ExceptionDialog.show_exception('Bad Pixels File Reader', exc)

    def get_widget(self) -> QWidget:
        return self._widget

    def set_num_bad_pixels(self, num_bad_pixels: int) -> None:
        self._line_edit.setText(str(num_bad_pixels))


class DetectorController:
    def __init__(
        self,
        settings: DetectorSettings,
        patterns_api: PatternsAPI,
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
        self._bit_depth_view_controller = SpinBoxParameterViewController(settings.bit_depth)
        self._bad_pixels_view_controller = BadPixelsViewController(
            settings.bad_pixels_file_path,
            settings.bad_pixels_file_type,
            patterns_api,
            file_dialog_factory,
        )

        layout = QFormLayout()
        layout.addRow('Detector Width [px]:', self._width_px_view_controller.get_widget())
        layout.addRow('Detector Height [px]:', self._height_px_view_controller.get_widget())
        layout.addRow('Pixel Width:', self._pixel_width_view_controller.get_widget())
        layout.addRow('Pixel Height:', self._pixel_height_view_controller.get_widget())
        layout.addRow('Bit Depth:', self._bit_depth_view_controller.get_widget())
        layout.addRow('Bad Pixels:', self._bad_pixels_view_controller.get_widget())
        view.setLayout(layout)

    def set_num_bad_pixels(self, num_bad_pixels: int) -> None:
        self._bad_pixels_view_controller.set_num_bad_pixels(num_bad_pixels)


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
        self._detector_controller = DetectorController(
            detector_settings, patterns_api, view.detector_view, file_dialog_factory
        )
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

        open_dataset_action = view.button_box.load_menu.addAction('Open File...')
        open_dataset_action.triggered.connect(self._wizard_controller.open_dataset)

        view.button_box.save_button.clicked.connect(self._save_dataset)
        view.button_box.close_button.clicked.connect(self._close_dataset)

        dataset_layout_action = view.button_box.analyze_menu.addAction('Dataset Layout...')
        dataset_layout_action.triggered.connect(self._show_dataset_layout)

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
            except Exception as exc:
                logger.exception(exc)
                ExceptionDialog.show_exception('File Writer', exc)

    def _show_dataset_layout(self) -> None:
        DatasetLayoutViewController.show_dialog(self._dataset, self._view)

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

    def handle_bad_pixels_changed(self, num_bad_pixels: int) -> None:
        self._detector_controller.set_num_bad_pixels(num_bad_pixels)

    def handle_array_inserted(self, index: int) -> None:
        self._tree_model.insert_array(index, self._dataset[index])

    def handle_array_changed(self, index: int) -> None:
        self._tree_model.refresh_array(index)

    def handle_dataset_reloaded(self) -> None:
        self._sync_model_to_view()

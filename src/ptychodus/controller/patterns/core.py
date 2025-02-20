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
from ...view.widgets import ExceptionDialog
from ..data import FileDialogFactory
from ..image import ImageController
from ..parametric import LengthWidgetParameterViewController, SpinBoxParameterViewController
from .dataset import DatasetTreeModel, DatasetTreeNode
from .info import PatternsInfoViewController
from .wizard import OpenDatasetWizardController

logger = logging.getLogger(__name__)


class DetectorController:
    def __init__(self, settings: DetectorSettings, view: DetectorView) -> None:
        self._widthInPixelsViewController = SpinBoxParameterViewController(settings.widthInPixels)
        self._heightInPixelsViewController = SpinBoxParameterViewController(settings.heightInPixels)
        self._pixelWidthViewController = LengthWidgetParameterViewController(
            settings.pixelWidthInMeters
        )
        self._pixelHeightViewController = LengthWidgetParameterViewController(
            settings.pixelHeightInMeters
        )
        self._bitDepthViewController = SpinBoxParameterViewController(settings.bitDepth)

        layout = QFormLayout()
        layout.addRow('Detector Width [px]:', self._widthInPixelsViewController.getWidget())
        layout.addRow('Detector Height [px]:', self._heightInPixelsViewController.getWidget())
        layout.addRow('Pixel Width:', self._pixelWidthViewController.getWidget())
        layout.addRow('Pixel Height:', self._pixelHeightViewController.getWidget())
        layout.addRow('Bit Depth:', self._bitDepthViewController.getWidget())
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
        self._detector_controller = DetectorController(detector_settings, view.detectorView)
        self._wizard_controller = OpenDatasetWizardController(
            pattern_settings,
            pattern_sizer,
            patterns_api,
            metadata_presenter,
            file_dialog_factory,
        )
        self._treeModel = DatasetTreeModel()

        view.treeView.setModel(self._treeModel)
        view.treeView.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        view.treeView.selectionModel().currentChanged.connect(self._updateView)
        self._updateView(QModelIndex(), QModelIndex())

        view.buttonBox.openButton.clicked.connect(self._wizard_controller.openDataset)
        view.buttonBox.saveButton.clicked.connect(self._saveDataset)
        view.buttonBox.infoButton.clicked.connect(self._openPatternsInfo)
        view.buttonBox.closeButton.clicked.connect(self._closeDataset)
        dataset.add_observer(self)

        self._syncModelToView()

    def _updateView(self, current: QModelIndex, previous: QModelIndex) -> None:
        if current.isValid():
            node = current.internalPointer()
            pixelGeometry = self._pattern_sizer.get_processed_pixel_geometry()
            self._image_controller.setArray(node.data, pixelGeometry)
        else:
            self._image_controller.clearArray()

    def _saveDataset(self) -> None:
        fileWriterChooser = self._patterns_api.getFileWriterChooser()
        filePath, nameFilter = self._file_dialog_factory.getSaveFilePath(
            self._view,
            'Save Diffraction File',
            nameFilters=fileWriterChooser.getDisplayNameList(),
            selectedNameFilter=fileWriterChooser.currentPlugin.displayName,
        )

        if filePath:
            try:
                self._patterns_api.savePatterns(filePath, nameFilter)
            except Exception as err:
                logger.exception(err)
                ExceptionDialog.showException('File Writer', err)

    def _openPatternsInfo(self) -> None:
        PatternsInfoViewController.showInfo(self._dataset, self._view)

    def _closeDataset(self) -> None:
        button = QMessageBox.question(
            self._view,
            'Confirm Close',
            'This will free the diffraction data from memory. Do you want to continue?',
        )

        if button != QMessageBox.StandardButton.Yes:
            return

        self._patterns_api.closePatterns()

    def _syncModelToView(self) -> None:
        rootNode = DatasetTreeNode.createRoot()

        for array in self._dataset:
            rootNode.createChild(array)

        self._treeModel.setRootNode(rootNode)

        info_text = self._dataset.get_info_text()
        self._view.infoLabel.setText(info_text)

    def handle_array_inserted(self, index: int) -> None:
        pass

    def handle_array_changed(self, index: int) -> None:
        pass

    def handle_dataset_reloaded(self) -> None:
        self._syncModelToView()

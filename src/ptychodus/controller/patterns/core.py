import logging

from PyQt5.QtCore import QModelIndex
from PyQt5.QtWidgets import QAbstractItemView, QFormLayout, QMessageBox

from ptychodus.api.observer import Observable, Observer

from ...model.patterns import (
    Detector,
    DiffractionDatasetInputOutputPresenter,
    DiffractionDatasetPresenter,
    DiffractionMetadataPresenter,
    DiffractionPatternPresenter,
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
    def __init__(self, detector: Detector, view: DetectorView) -> None:
        self._widthInPixelsViewController = SpinBoxParameterViewController(detector.widthInPixels)
        self._heightInPixelsViewController = SpinBoxParameterViewController(detector.heightInPixels)
        self._pixelWidthViewController = LengthWidgetParameterViewController(
            detector.pixelWidthInMeters
        )
        self._pixelHeightViewController = LengthWidgetParameterViewController(
            detector.pixelHeightInMeters
        )
        self._bitDepthViewController = SpinBoxParameterViewController(detector.bitDepth)

        layout = QFormLayout()
        layout.addRow('Detector Width [px]:', self._widthInPixelsViewController.getWidget())
        layout.addRow('Detector Height [px]:', self._heightInPixelsViewController.getWidget())
        layout.addRow('Pixel Width:', self._pixelWidthViewController.getWidget())
        layout.addRow('Pixel Height:', self._pixelHeightViewController.getWidget())
        layout.addRow('Bit Depth:', self._bitDepthViewController.getWidget())
        view.setLayout(layout)


class PatternsController(Observer):
    def __init__(
        self,
        detector: Detector,
        ioPresenter: DiffractionDatasetInputOutputPresenter,
        metadataPresenter: DiffractionMetadataPresenter,
        datasetPresenter: DiffractionDatasetPresenter,
        patternPresenter: DiffractionPatternPresenter,
        imageController: ImageController,
        view: PatternsView,
        fileDialogFactory: FileDialogFactory,
    ) -> None:
        super().__init__()
        self._detector = detector
        self._datasetPresenter = datasetPresenter
        self._ioPresenter = ioPresenter
        self._imageController = imageController
        self._view = view
        self._fileDialogFactory = fileDialogFactory
        self._detectorController = DetectorController(detector, view.detectorView)
        self._wizardController = OpenDatasetWizardController.createInstance(
            ioPresenter,
            metadataPresenter,
            datasetPresenter,
            patternPresenter,
            view.openDatasetWizard,
            fileDialogFactory,
        )
        self._treeModel = DatasetTreeModel()

        view.treeView.setModel(self._treeModel)
        view.treeView.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        view.treeView.selectionModel().currentChanged.connect(self._updateView)
        self._updateView(QModelIndex(), QModelIndex())

        view.buttonBox.openButton.clicked.connect(self._wizardController.openDataset)
        view.buttonBox.saveButton.clicked.connect(self._saveDataset)
        view.buttonBox.infoButton.clicked.connect(self._openPatternsInfo)
        view.buttonBox.closeButton.clicked.connect(self._closeDataset)
        view.buttonBox.closeButton.setEnabled(False)  # TODO
        datasetPresenter.addObserver(self)

        self._syncModelToView()

    def _updateView(self, current: QModelIndex, previous: QModelIndex) -> None:
        if current.isValid():
            node = current.internalPointer()
            pixelGeometry = self._detector.getPixelGeometry()
            self._imageController.setArray(node.data, pixelGeometry)
        else:
            self._imageController.clearArray()

    def _saveDataset(self) -> None:
        filePath, nameFilter = self._fileDialogFactory.getSaveFilePath(
            self._view,
            'Save Diffraction File',
            nameFilters=self._ioPresenter.getSaveFileFilterList(),
            selectedNameFilter=self._ioPresenter.getSaveFileFilter(),
        )

        if filePath:
            try:
                self._ioPresenter.saveDiffractionFile(filePath, nameFilter)
            except Exception as err:
                logger.exception(err)
                ExceptionDialog.showException('File Writer', err)

    def _openPatternsInfo(self) -> None:
        PatternsInfoViewController.showInfo(self._datasetPresenter, self._view)

    def _closeDataset(self) -> None:
        button = QMessageBox.question(
            self._view,
            'Confirm Close',
            'This will free the diffraction data from memory. Do you want to continue?',
        )

        if button != QMessageBox.StandardButton.Yes:
            return

        logger.error('Close not implemented!')  # TODO

    def _syncModelToView(self) -> None:
        rootNode = DatasetTreeNode.createRoot()

        for arrayPresenter in self._datasetPresenter:
            rootNode.createChild(arrayPresenter)

        self._treeModel.setRootNode(rootNode)

        infoText = self._datasetPresenter.getInfoText()
        self._view.infoLabel.setText(infoText)

    def update(self, observable: Observable) -> None:
        if observable is self._datasetPresenter:
            self._syncModelToView()

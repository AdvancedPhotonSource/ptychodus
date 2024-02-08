from __future__ import annotations
import logging

from PyQt5.QtCore import QItemSelection
from PyQt5.QtWidgets import QAbstractItemView, QMessageBox

from ...api.observer import Observable, Observer
from ...model.image import ImagePresenter
from ...model.patterns import (DetectorPresenter, DiffractionDatasetInputOutputPresenter,
                               DiffractionDatasetPresenter, DiffractionMetadataPresenter,
                               DiffractionPatternPresenter)
from ...view.image import ImageView
from ...view.patterns import PatternsView
from ..data import FileDialogFactory
from ..image import ImageController
from .detector import DetectorController
from .info import PatternsInfoViewController
from .treeModel import DatasetTreeModel, DatasetTreeNode
from .wizard import OpenDatasetWizardController

logger = logging.getLogger(__name__)


class PatternsController(Observer):

    def __init__(self, detectorPresenter: DetectorPresenter,
                 ioPresenter: DiffractionDatasetInputOutputPresenter,
                 metadataPresenter: DiffractionMetadataPresenter,
                 datasetPresenter: DiffractionDatasetPresenter,
                 patternPresenter: DiffractionPatternPresenter, imagePresenter: ImagePresenter,
                 view: PatternsView, imageView: ImageView,
                 fileDialogFactory: FileDialogFactory) -> None:
        super().__init__()
        self._detectorPresenter = detectorPresenter
        self._datasetPresenter = datasetPresenter
        self._ioPresenter = ioPresenter
        self._imagePresenter = imagePresenter
        self._view = view
        self._fileDialogFactory = fileDialogFactory
        self._detectorController = DetectorController.createInstance(detectorPresenter,
                                                                     view.detectorView)
        self._imageController = ImageController.createInstance(imagePresenter, imageView,
                                                               fileDialogFactory)
        self._wizardController = OpenDatasetWizardController.createInstance(
            ioPresenter, metadataPresenter, datasetPresenter, patternPresenter,
            view.openDatasetWizard, fileDialogFactory)
        self._treeModel = DatasetTreeModel()

    @classmethod
    def createInstance(cls, detectorPresenter: DetectorPresenter,
                       ioPresenter: DiffractionDatasetInputOutputPresenter,
                       metadataPresenter: DiffractionMetadataPresenter,
                       datasetPresenter: DiffractionDatasetPresenter,
                       patternPresenter: DiffractionPatternPresenter,
                       imagePresenter: ImagePresenter, view: PatternsView, imageView: ImageView,
                       fileDialogFactory: FileDialogFactory) -> PatternsController:
        controller = cls(detectorPresenter, ioPresenter, metadataPresenter, datasetPresenter,
                         patternPresenter, imagePresenter, view, imageView, fileDialogFactory)

        view.treeView.setModel(controller._treeModel)
        view.treeView.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        view.treeView.selectionModel().selectionChanged.connect(controller._updateView)
        view.buttonBox.openButton.clicked.connect(controller._wizardController.openDataset)
        view.buttonBox.saveButton.clicked.connect(controller._saveDataset)
        view.buttonBox.infoButton.clicked.connect(controller._openPatternsInfo)
        view.buttonBox.closeButton.clicked.connect(controller._closeDataset)
        view.buttonBox.closeButton.setEnabled(False)  # TODO
        datasetPresenter.addObserver(controller)

        controller._syncModelToView()

        return controller

    def _updateView(self, selected: QItemSelection, deselected: QItemSelection) -> None:
        for index in deselected.indexes():
            self._imagePresenter.clearArray()
            break

        for index in selected.indexes():
            node = index.internalPointer()
            pixelGeometry = self._detectorPresenter.getPixelGeometry()
            self._imagePresenter.setArray(node.data, pixelGeometry)
            break

    def _saveDataset(self) -> None:
        filePath, nameFilter = self._fileDialogFactory.getSaveFilePath(
            self._view,
            'Save Diffraction File',
            nameFilters=self._ioPresenter.getSaveFileFilterList(),
            selectedNameFilter=self._ioPresenter.getSaveFileFilter())

        if filePath:
            self._ioPresenter.saveDiffractionFile(filePath)

    def _openPatternsInfo(self) -> None:
        PatternsInfoViewController.showInfo(self._datasetPresenter, self._view)

    def _closeDataset(self) -> None:
        button = QMessageBox.question(
            self._view, 'Confirm Close',
            'This will free the diffraction data from memory. Do you want to continue?')

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

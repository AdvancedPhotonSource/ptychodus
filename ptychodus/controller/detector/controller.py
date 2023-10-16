from __future__ import annotations
import logging

from PyQt5.QtCore import QItemSelection
from PyQt5.QtWidgets import QAbstractItemView, QMessageBox

from ...api.observer import Observable, Observer
from ...model import DetectorPresenter, MetadataPresenter
from ...model.data import (DiffractionDatasetInputOutputPresenter, DiffractionDatasetPresenter,
                           DiffractionPatternPresenter)
from ...model.image import ImagePresenter
from ...model.probe import ApparatusPresenter
from ...view.detector import DetectorView
from ...view.image import ImageView
from ..data import FileDialogFactory
from ..image import ImageController
from .inspect import InspectDatasetController
from .parameters import DetectorParametersController
from .treeModel import DatasetTreeModel, DatasetTreeNode
from .wizard import OpenDatasetWizardController

logger = logging.getLogger(__name__)


class DetectorController(Observer):

    def __init__(self, detectorPresenter: DetectorPresenter,
                 apparatusPresenter: ApparatusPresenter,
                 ioPresenter: DiffractionDatasetInputOutputPresenter,
                 metadataPresenter: MetadataPresenter,
                 datasetPresenter: DiffractionDatasetPresenter,
                 patternPresenter: DiffractionPatternPresenter, imagePresenter: ImagePresenter,
                 view: DetectorView, imageView: ImageView,
                 fileDialogFactory: FileDialogFactory) -> None:
        super().__init__()
        self._datasetPresenter = datasetPresenter
        self._ioPresenter = ioPresenter
        self._imagePresenter = imagePresenter
        self._view = view
        self._fileDialogFactory = fileDialogFactory
        self._imageController = ImageController.createInstance(imagePresenter, imageView,
                                                               fileDialogFactory)
        self._parametersController = DetectorParametersController.createInstance(
            detectorPresenter, apparatusPresenter, view.parametersView)
        self._wizardController = OpenDatasetWizardController.createInstance(
            ioPresenter, metadataPresenter, datasetPresenter, patternPresenter,
            view.dataView.openDatasetWizard, fileDialogFactory)
        self._inspectDatasetController = InspectDatasetController.createInstance(
            datasetPresenter, view.dataView.inspectDatasetDialog)
        self._treeModel = DatasetTreeModel()

    @classmethod
    def createInstance(cls, detectorPresenter: DetectorPresenter,
                       apparatusPresenter: ApparatusPresenter,
                       ioPresenter: DiffractionDatasetInputOutputPresenter,
                       metadataPresenter: MetadataPresenter,
                       datasetPresenter: DiffractionDatasetPresenter,
                       patternPresenter: DiffractionPatternPresenter,
                       imagePresenter: ImagePresenter, view: DetectorView, imageView: ImageView,
                       fileDialogFactory: FileDialogFactory) -> DetectorController:
        controller = cls(detectorPresenter, apparatusPresenter, ioPresenter, metadataPresenter,
                         datasetPresenter, patternPresenter, imagePresenter, view, imageView,
                         fileDialogFactory)

        view.dataView.treeView.setModel(controller._treeModel)
        view.dataView.treeView.setSelectionBehavior(QAbstractItemView.SelectRows)
        view.dataView.treeView.selectionModel().selectionChanged.connect(controller._updateView)
        view.dataView.buttonBox.openButton.clicked.connect(
            controller._wizardController.openDataset)
        view.dataView.buttonBox.saveButton.clicked.connect(controller._saveDataset)
        view.dataView.buttonBox.inspectButton.clicked.connect(
            controller._inspectDatasetController.inspectDataset)
        view.dataView.buttonBox.closeButton.clicked.connect(controller._closeDataset)
        datasetPresenter.addObserver(controller)

        controller._syncModelToView()

        return controller

    def _updateView(self, selected: QItemSelection, deselected: QItemSelection) -> None:
        for index in deselected.indexes():
            self._imagePresenter.clearArray()
            break

        for index in selected.indexes():
            node = index.internalPointer()
            self._imagePresenter.setArray(node.data)
            break

    def _saveDataset(self) -> None:
        filePath, nameFilter = self._fileDialogFactory.getSaveFilePath(
            self._view,
            'Save Diffraction File',
            nameFilters=self._ioPresenter.getSaveFileFilterList(),
            selectedNameFilter=self._ioPresenter.getSaveFileFilter())

        if filePath:
            self._ioPresenter.saveDiffractionFile(filePath)

    def _closeDataset(self) -> None:
        button = QMessageBox.question(
            self._view, 'Confirm Close',
            'This will free the diffraction data from memory. Do you want to continue?')

        if button != QMessageBox.Yes:
            return

        logger.error('Close not implemented!')  # FIXME

    def _syncModelToView(self) -> None:
        rootNode = DatasetTreeNode.createRoot()

        for arrayPresenter in self._datasetPresenter:
            rootNode.createChild(arrayPresenter)

        self._treeModel.setRootNode(rootNode)

        infoText = self._datasetPresenter.getInfoText()
        self._view.dataView.infoLabel.setText(infoText)

    def update(self, observable: Observable) -> None:
        if observable is self._datasetPresenter:
            self._syncModelToView()

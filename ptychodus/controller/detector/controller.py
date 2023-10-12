from __future__ import annotations
import logging

from PyQt5.QtCore import QItemSelection
from PyQt5.QtWidgets import QAbstractItemView

from ...api.observer import Observable, Observer
from ...model import DetectorPresenter
from ...model.data import DiffractionDatasetInputOutputPresenter, DiffractionDatasetPresenter
from ...model.image import ImagePresenter
from ...model.probe import ApparatusPresenter
from ...view.detector import DetectorView
from ...view.image import ImageView
from ..data import FileDialogFactory
from ..image import ImageController
from .parameters import DetectorParametersController
from .treeModel import DatasetTreeModel, DatasetTreeNode

logger = logging.getLogger(__name__)


class DetectorController(Observer):

    def __init__(self, detectorPresenter: DetectorPresenter,
                 apparatusPresenter: ApparatusPresenter,
                 inputOutputPresenter: DiffractionDatasetInputOutputPresenter,
                 datasetPresenter: DiffractionDatasetPresenter, imagePresenter: ImagePresenter,
                 view: DetectorView, imageView: ImageView,
                 fileDialogFactory: FileDialogFactory) -> None:
        super().__init__()
        self._inputOutputPresenter = inputOutputPresenter
        self._datasetPresenter = datasetPresenter
        self._imagePresenter = imagePresenter
        self._view = view
        self._imageView = imageView
        self._fileDialogFactory = fileDialogFactory
        self._imageController = ImageController.createInstance(imagePresenter, imageView,
                                                               fileDialogFactory)
        self._parametersController = DetectorParametersController.createInstance(
            detectorPresenter, apparatusPresenter, view.parametersView)
        self._treeModel = DatasetTreeModel()

    @classmethod
    def createInstance(cls, detectorPresenter: DetectorPresenter,
                       apparatusPresenter: ApparatusPresenter,
                       inputOutputPresenter: DiffractionDatasetInputOutputPresenter,
                       datasetPresenter: DiffractionDatasetPresenter,
                       imagePresenter: ImagePresenter, view: DetectorView, imageView: ImageView,
                       fileDialogFactory: FileDialogFactory) -> DetectorController:
        controller = cls(detectorPresenter, apparatusPresenter, inputOutputPresenter,
                         datasetPresenter, imagePresenter, view, imageView, fileDialogFactory)

        view.dataView.treeView.setModel(controller._treeModel)
        view.dataView.treeView.setSelectionBehavior(QAbstractItemView.SelectRows)
        view.dataView.treeView.selectionModel().selectionChanged.connect(controller._updateView)
        view.dataView.buttonBox.openButton.clicked.connect(controller._openDiffractionFile)
        view.dataView.buttonBox.saveButton.clicked.connect(controller._saveDiffractionFile)
        view.dataView.buttonBox.inspectButton.clicked.connect(controller._inspectDiffractionFile)
        view.dataView.buttonBox.closeButton.clicked.connect(controller._closeDiffractionFile)
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

    def _openDiffractionFile(self) -> None:
        self._view.dataView.openDataWizard.restart()
        self._view.dataView.openDataWizard.show()

    def _saveDiffractionFile(self) -> None:
        filePath, nameFilter = self._fileDialogFactory.getSaveFilePath(
            self._view,
            'Save Diffraction File',
            nameFilters=self._inputOutputPresenter.getSaveFileFilterList(),
            selectedNameFilter=self._inputOutputPresenter.getSaveFileFilter())

        if filePath:
            self._inputOutputPresenter.saveDiffractionFile(filePath)

    def _inspectDiffractionFile(self) -> None:
        self._view.dataView.inspectDataDialog.show()

    def _closeDiffractionFile(self) -> None:
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

from __future__ import annotations
import logging

from PyQt5.QtCore import QItemSelection
from PyQt5.QtWidgets import QAbstractItemView

from ...api.observer import Observable, Observer
from ...model import DetectorPresenter
from ...model.data import DiffractionDatasetPresenter
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
                 datasetPresenter: DiffractionDatasetPresenter, imagePresenter: ImagePresenter,
                 view: DetectorView, imageView: ImageView,
                 fileDialogFactory: FileDialogFactory) -> None:
        super().__init__()
        self._datasetPresenter = datasetPresenter
        self._imagePresenter = imagePresenter
        self._view = view
        self._imageView = imageView
        self._imageController = ImageController.createInstance(imagePresenter, imageView,
                                                               fileDialogFactory)
        self._parametersController = DetectorParametersController.createInstance(
            detectorPresenter, apparatusPresenter, view.parametersView)
        self._treeModel = DatasetTreeModel()

    @classmethod
    def createInstance(cls, detectorPresenter: DetectorPresenter,
                       apparatusPresenter: ApparatusPresenter,
                       datasetPresenter: DiffractionDatasetPresenter,
                       imagePresenter: ImagePresenter, view: DetectorView, imageView: ImageView,
                       fileDialogFactory: FileDialogFactory) -> DetectorController:
        controller = cls(detectorPresenter, apparatusPresenter, datasetPresenter, imagePresenter,
                         view, imageView, fileDialogFactory)

        view.dataView.treeView.setModel(controller._treeModel)
        view.dataView.treeView.setSelectionBehavior(QAbstractItemView.SelectRows)
        view.dataView.treeView.selectionModel().selectionChanged.connect(controller._updateView)
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

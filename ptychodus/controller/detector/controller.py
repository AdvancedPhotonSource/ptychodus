from __future__ import annotations
import logging

from ...api.observer import Observable, Observer
from ...model.data import DiffractionDatasetPresenter
from ...model.image import ImagePresenter
from ...view import DetectorView, ImageView
from ..data import FileDialogFactory
from ..image import ImageController
from .treeModel import DatasetTreeModel, DatasetTreeNode

logger = logging.getLogger(__name__)


class DetectorController(Observer):

    def __init__(self, datasetPresenter: DiffractionDatasetPresenter,
                 imagePresenter: ImagePresenter, view: DetectorView, imageView: ImageView,
                 fileDialogFactory: FileDialogFactory) -> None:
        super().__init__()
        self._datasetPresenter = datasetPresenter
        self._imagePresenter = imagePresenter
        self._view = view
        self._imageView = imageView
        self._imageController = ImageController.createInstance(imagePresenter, imageView,
                                                               fileDialogFactory)
        self._treeModel = DatasetTreeModel()

    @classmethod
    def createInstance(cls, datasetPresenter: DiffractionDatasetPresenter,
                       imagePresenter: ImagePresenter, view: DetectorView, imageView: ImageView,
                       fileDialogFactory: FileDialogFactory) -> DetectorController:
        controller = cls(datasetPresenter, imagePresenter, view, imageView, fileDialogFactory)

        imageView.imageRibbon.indexGroupBox.setVisible(False)
        view.dataView.treeView.setModel(controller._treeModel)
        datasetPresenter.addObserver(controller)

        #view.treeView.selectionModel().currentChanged.connect(
        #    controller._updateCurrentPatternIndex)  # FIXME do same for object/probe controllers

        #controller._syncModelToView()

        return controller

    #def _updateCurrentPatternIndex(self, index: QModelIndex) -> None:
    #    self._activePatternPresenter.setCurrentPatternIndex(index.row())

    #def _updateSelection(self) -> None:
    #    row = self._activePatternPresenter.getCurrentPatternIndex()
    #    index = self._treeModel.index(row, 0)
    #    self._view.treeView.setCurrentIndex(index)

    #def setCurrentPatternIndex(self, index: int) -> None:
    #    try:
    #        data = self._dataset[index]
    #    except IndexError:
    #        logger.exception('Invalid data index!')
    #        return

    #    self._array.removeObserver(self)
    #    self._array = data
    #    self._array.addObserver(self)
    #    self.notifyObservers()

    #def getCurrentPatternIndex(self) -> int:
    #    return self._array.getIndex()

    #def getNumberOfImages(self) -> int:
    #    return self._array.getData().shape[0]

    #def getImage(self, index: int) -> DiffractionPatternData:
    #    return self._array.getData()[index]

    #def _renderImageData(self, index: int) -> None:
    #    array = self._activePatternPresenter.getImage(index)
    #    self._imagePresenter.setArray(array)

    def _syncModelToView(self) -> None:
        rootNode = DatasetTreeNode.createRoot()

        for arrayPresenter in self._datasetPresenter:
            arrayNode = rootNode.createChild(arrayPresenter)

        self._treeModel.setRootNode(rootNode)

        # FIXME self._renderImageData(index)

    def update(self, observable: Observable) -> None:
        if observable is self._datasetPresenter:
            self._syncModelToView()

        #if observable is self._activePatternPresenter:
        #    self._syncModelToView()
        #elif observable is self._activePatternPresenter:
        #    self._updateSelection()
        #elif observable is self._dataset:
        #    self._array.removeObserver(self)
        #    self._array = SimpleDiffractionPatternArray.createNullInstance()
        #    self.notifyObservers()
        #elif observable is self._array:
        #    self.notifyObservers()

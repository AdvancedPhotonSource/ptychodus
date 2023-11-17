from __future__ import annotations

from ...api.observer import Observable, Observer
from ...model.image import ImagePresenter
from ...model.object import ObjectPresenter
from ...model.probe import ApparatusPresenter
from ...view.image import ImageView
from ..data import FileDialogFactory
from ..image import ImageController


class ObjectImageController(Observer):

    def __init__(self, apparatusPresenter: ApparatusPresenter, presenter: ObjectPresenter,
                 imagePresenter: ImagePresenter, imageView: ImageView,
                 fileDialogFactory: FileDialogFactory) -> None:
        super().__init__()
        self._apparatusPresenter = apparatusPresenter
        self._presenter = presenter
        self._imagePresenter = imagePresenter
        self._imageView = imageView
        self._imageController = ImageController.createInstance(imagePresenter, imageView,
                                                               fileDialogFactory)

    @classmethod
    def createInstance(cls, apparatusPresenter: ApparatusPresenter, presenter: ObjectPresenter,
                       imagePresenter: ImagePresenter, imageView: ImageView,
                       fileDialogFactory: FileDialogFactory) -> ObjectImageController:
        controller = cls(apparatusPresenter, presenter, imagePresenter, imageView,
                         fileDialogFactory)
        apparatusPresenter.addObserver(controller)
        presenter.addObserver(controller)
        controller._syncModelToView()
        return controller

    def _syncModelToView(self) -> None:
        array = self._presenter.getSelectedObjectFlattenedArray()

        if array is None:
            self._imagePresenter.clearArray()
        else:
            pixelGeometry = self._apparatusPresenter.getObjectPlanePixelGeometry()
            self._imagePresenter.setArray(array, pixelGeometry)

    def update(self, observable: Observable) -> None:
        if observable is self._apparatusPresenter:
            self._syncModelToView()
        elif observable is self._presenter:
            self._syncModelToView()

from __future__ import annotations

from ...api.observer import Observable, Observer
from ...model.image import ImagePresenter
from ...model.object import ObjectPresenter
from ...view import ImageView
from ..data import FileDialogFactory
from ..image import ImageController


class ObjectImageController(Observer):

    def __init__(self, presenter: ObjectPresenter, imagePresenter: ImagePresenter,
                 imageView: ImageView, fileDialogFactory: FileDialogFactory) -> None:
        super().__init__()
        self._presenter = presenter
        self._imagePresenter = imagePresenter
        self._imageView = imageView
        self._imageController = ImageController.createInstance(imagePresenter, imageView,
                                                               fileDialogFactory)

    @classmethod
    def createInstance(cls, presenter: ObjectPresenter, imagePresenter: ImagePresenter,
                       imageView: ImageView,
                       fileDialogFactory: FileDialogFactory) -> ObjectImageController:
        controller = cls(presenter, imagePresenter, imageView, fileDialogFactory)
        presenter.addObserver(controller)
        controller._syncModelToView()
        imageView.imageRibbon.indexGroupBox.setVisible(False)
        return controller

    def _syncModelToView(self) -> None:
        array = self._presenter.getSelectedObjectArray()
        self._imagePresenter.setArray(array)

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()

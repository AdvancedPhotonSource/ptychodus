from __future__ import annotations

from ...api.observer import Observable, Observer
from ...model.image import ImagePresenter
from ...model.object import ObjectPresenter
from ...view import ImageView
from ..data import FileDialogFactory
from ..image import ImageController


class ObjectImageController(Observer):

    def __init__(self, presenter: ObjectPresenter, imagePresenter: ImagePresenter, view: ImageView,
                 fileDialogFactory: FileDialogFactory) -> None:
        super().__init__()
        self._presenter = presenter
        self._imagePresenter = imagePresenter
        self._view = view
        self._imageController = ImageController.createInstance(imagePresenter, view,
                                                               fileDialogFactory)

    @classmethod
    def createInstance(cls, presenter: ObjectPresenter, imagePresenter: ImagePresenter,
                       view: ImageView,
                       fileDialogFactory: FileDialogFactory) -> ObjectImageController:
        controller = cls(presenter, imagePresenter, view, fileDialogFactory)
        presenter.addObserver(controller)
        controller._syncModelToView()
        view.imageRibbon.indexGroupBox.setVisible(False)
        return controller

    def _syncModelToView(self) -> None:
        array = self._presenter.getObject()
        self._imagePresenter.setArray(array)

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()

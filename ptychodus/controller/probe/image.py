from __future__ import annotations

from ...api.observer import Observable, Observer
from ...model.image import ImagePresenter
from ...model.probe import ProbePresenter
from ...view.image import ImageView
from ..data import FileDialogFactory
from ..image import ImageController


class ProbeImageController(Observer):

    def __init__(self, presenter: ProbePresenter, imagePresenter: ImagePresenter,
                 imageView: ImageView, fileDialogFactory: FileDialogFactory) -> None:
        super().__init__()
        self._presenter = presenter
        self._imagePresenter = imagePresenter
        self._imageView = imageView
        self._imageController = ImageController.createInstance(imagePresenter, imageView,
                                                               fileDialogFactory)

    @classmethod
    def createInstance(cls, presenter: ProbePresenter, imagePresenter: ImagePresenter,
                       imageView: ImageView,
                       fileDialogFactory: FileDialogFactory) -> ProbeImageController:
        controller = cls(presenter, imagePresenter, imageView, fileDialogFactory)
        presenter.addObserver(controller)
        controller._syncModelToView()
        return controller

    def _syncModelToView(self) -> None:
        array = self._presenter.getSelectedProbeFlattenedArray()

        if array is None:
            self._imagePresenter.clearArray()
        else:
            self._imagePresenter.setArray(array)

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()

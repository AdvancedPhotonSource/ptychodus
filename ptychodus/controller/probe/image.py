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
        imageView.imageRibbon.indexGroupBox.setTitle('Probe Mode')
        imageView.imageRibbon.indexGroupBox.indexSpinBox.valueChanged.connect(
            controller._renderImageData)
        return controller

    def _renderImageData(self, index: int) -> None:
        array = self._presenter.getSelectedProbeModeArray(index)

        if array is None:
            self._imagePresenter.clearArray()
        else:
            self._imagePresenter.setArray(array)

    def _syncModelToView(self) -> None:
        # FIXME what to do with monitor screen?
        numberOfProbeModes = self._presenter.getNumberOfProbeModes()
        self._imageView.imageRibbon.indexGroupBox.indexSpinBox.setEnabled(numberOfProbeModes > 0)
        self._imageView.imageRibbon.indexGroupBox.indexSpinBox.setRange(0, numberOfProbeModes - 1)

        index = self._imageView.imageRibbon.indexGroupBox.indexSpinBox.value()
        self._renderImageData(index)

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()

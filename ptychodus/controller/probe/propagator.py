from PyQt5.QtWidgets import QStatusBar, QWidget

from ...model.analysis import ProbePropagator
from ...model.image import ImagePresenter
from ...view.probe import ProbePropagationDialog
from ..data import FileDialogFactory
from ..image import ImageController


class ProbePropagationViewController:

    def __init__(self, propagator: ProbePropagator, imagePresenter: ImagePresenter,
                 fileDialogFactory: FileDialogFactory, statusBar: QStatusBar,
                 parent: QWidget | None) -> None:
        super().__init__()
        self._propagator = propagator
        self._dialog = ProbePropagationDialog.createInstance(statusBar, parent)
        self._imageController = ImageController.createInstance(imagePresenter,
                                                               self._dialog.xyView.imageView,
                                                               fileDialogFactory)

    def propagate(self, itemIndex: int) -> None:
        # FIXME include item name in window title
        _ = self._propagator.propagate(itemIndex)  # FIXME start, stop, count
        # FIXME do something with result
        self._dialog.open()

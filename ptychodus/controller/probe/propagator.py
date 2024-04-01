import logging

from PyQt5.QtWidgets import QStatusBar, QWidget

from ...model.analysis import ProbePropagator
from ...model.visualization import VisualizationEngine
from ...view.probe import ProbePropagationDialog
from ..data import FileDialogFactory

logger = logging.getLogger(__name__)


class ProbePropagationViewController:

    def __init__(self, propagator: ProbePropagator, engine: VisualizationEngine,
                 fileDialogFactory: FileDialogFactory, statusBar: QStatusBar,
                 parent: QWidget | None) -> None:
        super().__init__()
        self._propagator = propagator
        self._dialog = ProbePropagationDialog.createInstance(parent)
        # FIXME self._imageController = ImageController.createInstance(engine, self._dialog.xyView.imageView, fileDialogFactory)

    def propagate(self, itemIndex: int) -> None:
        itemName = self._propagator.getName(itemIndex)
        _ = self._propagator.propagate(itemIndex)  # FIXME start, stop, count
        # FIXME do something with result
        self._dialog.setWindowTitle(f'Propagate Probe: {itemName}')
        self._dialog.open()

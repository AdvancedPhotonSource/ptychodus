from PyQt5.QtWidgets import QStatusBar, QWidget

from ...model.analysis import ProbePropagator
from ...view.probe import ProbePropagationDialog
from .listModel import ProbeListModel


class ProbePropagationViewController:

    def __init__(self, propagator: ProbePropagator, listModel: ProbeListModel,
                 statusBar: QStatusBar, parent: QWidget | None) -> None:
        super().__init__()
        self._propagator = propagator
        self._dialog = ProbePropagationDialog.createInstance(statusBar, parent)

    def propagate(self, itemIndex: int) -> None:
        _ = self._propagator.propagate(itemIndex)
        # FIXME do something with result
        self._dialog.open()

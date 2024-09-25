from PyQt5.QtWidgets import QWidget

from ...model.ptychopack import PtychoPackReconstructorLibrary
from ...view.ptychopack import PtychoPackAlgorithm, PtychoPackView
from ..reconstructor import ReconstructorViewControllerFactory
from .core import PtychoPackController


class PtychoPackViewControllerFactory(ReconstructorViewControllerFactory):

    def __init__(self, model: PtychoPackReconstructorLibrary) -> None:
        super().__init__()
        self._model = model
        self._controllerList: list[PtychoPackController] = list()

    @property
    def backendName(self) -> str:
        return "PtychoPack"

    def createViewController(self, reconstructorName: str) -> QWidget:
        if reconstructorName.casefold() == "dm":
            view = PtychoPackView(PtychoPackAlgorithm.DM)
        elif reconstructorName.casefold() == "raar":
            view = PtychoPackView(PtychoPackAlgorithm.RAAR)
        else:
            view = PtychoPackView(PtychoPackAlgorithm.PIE)

        controller = PtychoPackController(self._model, view)
        self._controllerList.append(controller)
        return view

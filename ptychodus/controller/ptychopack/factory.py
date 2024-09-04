from PyQt5.QtWidgets import QWidget

from ...model.ptychopack import PtychoPackReconstructorLibrary
from ...view.ptychopack import PtychoPackParametersView
from ..reconstructor import ReconstructorViewControllerFactory
from .controller import PtychoPackParametersController


class PtychoPackViewControllerFactory(ReconstructorViewControllerFactory):

    def __init__(self, model: PtychoPackReconstructorLibrary) -> None:
        super().__init__()
        self._model = model
        self._controllerList: list[PtychoPackParametersController] = list()

    @property
    def backendName(self) -> str:
        return 'PtychoPack'

    def createViewController(self, reconstructorName: str) -> QWidget:
        view = PtychoPackParametersView()
        controller = PtychoPackParametersController(self._model, view)
        self._controllerList.append(controller)
        return view

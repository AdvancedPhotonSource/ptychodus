from PyQt5.QtWidgets import QWidget

from ...model.tike import TikeReconstructorLibrary
from ...view.tike import TikeParametersView
from ..reconstructor import ReconstructorViewControllerFactory
from .controller import TikeParametersController


class TikeViewControllerFactory(ReconstructorViewControllerFactory):

    def __init__(self, model: TikeReconstructorLibrary) -> None:
        super().__init__()
        self._model = model
        self._controllerList: list[TikeParametersController] = list()

    @property
    def backendName(self) -> str:
        return "Tike"

    def createViewController(self, reconstructorName: str) -> QWidget:
        view = None

        if reconstructorName == "rpie":
            view = TikeParametersView.createInstance(showAlpha=True, showStepLength=False)
        elif reconstructorName == "lstsq_grad":
            view = TikeParametersView.createInstance(showAlpha=False, showStepLength=False)
        elif reconstructorName == "dm":
            view = TikeParametersView.createInstance(showAlpha=False, showStepLength=False)
        else:
            view = TikeParametersView.createInstance(showAlpha=True, showStepLength=True)

        controller = TikeParametersController.createInstance(self._model, view)
        self._controllerList.append(controller)

        return view

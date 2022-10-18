from __future__ import annotations

from PyQt5.QtWidgets import QWidget

from ..api.observer import Observable, Observer
from ..model import PtychoNNReconstructorLibrary, PtychoNNPresenter
from ..view import PtychoNNParametersView, PtychoNNBasicParametersView
from .reconstructor import ReconstructorViewControllerFactory


class PtychoNNParametersController:
    pass  # TODO


class PtychoNNViewControllerFactory(ReconstructorViewControllerFactory):

    def __init__(self, model: PtychoNNReconstructorLibrary) -> None:
        super().__init__()
        self._model = model
        self._controllerList: list[PtychoNNParametersController] = list()

    @property
    def backendName(self) -> str:
        return 'PtychoNN'

    def createViewController(self, reconstructorName: str) -> QWidget:
        view = PtychoNNParametersView.createInstance()

        # TODO controller = PtychoNNParametersController.createInstance(self._model, view)
        # TODO self._controllerList.append(controller)

        return view

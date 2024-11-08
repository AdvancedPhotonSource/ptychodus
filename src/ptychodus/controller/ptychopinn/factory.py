from ..reconstructor import ReconstructorViewControllerFactory

from PyQt5.QtWidgets import QWidget

from ptychodus.model.ptychopinn.core import PtychoPINNReconstructorLibrary
from ...view.ptychopinn import PtychoPINNParametersView
from ..data import FileDialogFactory
from .controller import PtychoPINNParametersController


class PtychoPINNViewControllerFactory(ReconstructorViewControllerFactory):
    def __init__(
        self, model: PtychoPINNReconstructorLibrary, fileDialogFactory: FileDialogFactory
    ) -> None:
        super().__init__()
        self._model = model
        self._fileDialogFactory = fileDialogFactory
        self._controllerList: list[PtychoPINNParametersController] = list()

    @property
    def backendName(self) -> str:
        return 'PtychoPINN'

    def createViewController(self, reconstructorName: str) -> QWidget:
        view = PtychoPINNParametersView.createInstance()

        controller = PtychoPINNParametersController.createInstance(
            self._model.modelPresenter, self._model.trainingPresenter, view, self._fileDialogFactory
        )
        self._controllerList.append(controller)

        return view

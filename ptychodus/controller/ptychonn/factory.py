from ..reconstructor import ReconstructorViewControllerFactory

from PyQt5.QtWidgets import QWidget

from ...model.ptychonn import PtychoNNReconstructorLibrary
from ...view.ptychonn import PtychoNNParametersView
from ..data import FileDialogFactory
from .controller import PtychoNNParametersController


class PtychoNNViewControllerFactory(ReconstructorViewControllerFactory):

    def __init__(self, model: PtychoNNReconstructorLibrary,
                 fileDialogFactory: FileDialogFactory) -> None:
        super().__init__()
        self._model = model
        self._fileDialogFactory = fileDialogFactory
        self._controllerList: list[PtychoNNParametersController] = list()

    @property
    def backendName(self) -> str:
        return 'PtychoNN'

    def createViewController(self, reconstructorName: str) -> QWidget:
        view = PtychoNNParametersView.createInstance()

        controller = PtychoNNParametersController.createInstance(self._model.modelPresenter,
                                                                 self._model.trainingPresenter,
                                                                 view, self._fileDialogFactory)
        self._controllerList.append(controller)

        return view

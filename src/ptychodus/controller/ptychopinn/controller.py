from __future__ import annotations

from ptychodus.model.ptychopinn.core import PtychoPINNModelPresenter, PtychoPINNTrainingPresenter
from ...view.ptychopinn import PtychoPINNParametersView
from ..data import FileDialogFactory
from .model import PtychoPINNModelParametersController
from .training import PtychoPINNTrainingParametersController


class PtychoPINNParametersController:

    def __init__(self, modelPresenter: PtychoPINNModelPresenter,
                 trainingPresenter: PtychoPINNTrainingPresenter, view: PtychoPINNParametersView,
                 fileDialogFactory: FileDialogFactory) -> None:
        super().__init__()
        self._modelParametersController = PtychoPINNModelParametersController.createInstance(
            modelPresenter, view.modelParametersView, fileDialogFactory)
        self._trainingParametersController = PtychoPINNTrainingParametersController.createInstance(
            trainingPresenter, view.trainingParametersView, fileDialogFactory)

    @classmethod
    def createInstance(cls, modelPresenter: PtychoPINNModelPresenter,
                       trainingPresenter: PtychoPINNTrainingPresenter,
                       view: PtychoPINNParametersView,
                       fileDialogFactory: FileDialogFactory) -> PtychoPINNParametersController:
        return cls(modelPresenter, trainingPresenter, view, fileDialogFactory)

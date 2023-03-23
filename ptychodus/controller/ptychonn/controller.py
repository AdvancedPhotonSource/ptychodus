from __future__ import annotations

from ...model.ptychonn import (PtychoNNModelPresenter, PtychoNNReconstructorLibrary,
                               PtychoNNTrainingPresenter)
from ...view import PtychoNNParametersView
from ..data import FileDialogFactory
from .model import PtychoNNModelParametersController
from .training import PtychoNNTrainingDataController, PtychoNNTrainingParametersController


class PtychoNNParametersController:

    def __init__(self, modelPresenter: PtychoNNModelPresenter,
                 trainingPresenter: PtychoNNTrainingPresenter, view: PtychoNNParametersView,
                 fileDialogFactory: FileDialogFactory) -> None:
        super().__init__()
        self._modelParametersController = PtychoNNModelParametersController.createInstance(
            modelPresenter, view.modelParametersView, fileDialogFactory)
        self._trainingParametersController = PtychoNNTrainingParametersController.createInstance(
            trainingPresenter, view.trainingParametersView, fileDialogFactory)
        self._trainingDataController = PtychoNNTrainingDataController.createInstance(
            trainingPresenter, view.trainingDataView, fileDialogFactory)

    @classmethod
    def createInstance(cls, modelPresenter: PtychoNNModelPresenter,
                       trainingPresenter: PtychoNNTrainingPresenter, view: PtychoNNParametersView,
                       fileDialogFactory: FileDialogFactory) -> PtychoNNParametersController:
        return cls(modelPresenter, trainingPresenter, view, fileDialogFactory)

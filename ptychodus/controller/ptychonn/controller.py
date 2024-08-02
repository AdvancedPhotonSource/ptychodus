from __future__ import annotations

from ...model.ptychonn import PtychoNNModelPresenter, PtychoNNTrainingPresenter, PtychoNNPositionPredictionPresenter
from ...view.ptychonn import PtychoNNParametersView
from ..data import FileDialogFactory
from .model import PtychoNNModelParametersController
from .training import PtychoNNTrainingParametersController
from .position import PtychoNNPositionPredictionParametersController

class PtychoNNParametersController:

    def __init__(self, modelPresenter: PtychoNNModelPresenter,
                 trainingPresenter: PtychoNNTrainingPresenter, 
                 positionPredictionPresenter: PtychoNNPositionPredictionPresenter,
                 view: PtychoNNParametersView,
                 fileDialogFactory: FileDialogFactory) -> None:
        super().__init__()
        self._modelParametersController = PtychoNNModelParametersController.createInstance(
            modelPresenter, view.modelParametersView, fileDialogFactory)
        self._trainingParametersController = PtychoNNTrainingParametersController.createInstance(
            trainingPresenter, view.trainingParametersView, fileDialogFactory)
        self._positionPredictionParamtersController = PtychoNNPositionPredictionParametersController.createInstance(
            positionPredictionPresenter, view.positionPredictionParametersView, fileDialogFactory)

    @classmethod
    def createInstance(cls, modelPresenter: PtychoNNModelPresenter,
                       trainingPresenter: PtychoNNTrainingPresenter, 
                       positionPredictionPresenter: PtychoNNPositionPredictionPresenter,
                       view: PtychoNNParametersView,
                       fileDialogFactory: FileDialogFactory) -> PtychoNNParametersController:
        return cls(modelPresenter, trainingPresenter, positionPredictionPresenter, view, fileDialogFactory)

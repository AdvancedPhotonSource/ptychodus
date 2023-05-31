from __future__ import annotations

from ...model.tike import TikeReconstructorLibrary
from ...view.tike import TikeParametersView
from .basic import TikeBasicParametersController
from .multigrid import TikeMultigridController
from .objectCorrection import TikeObjectCorrectionController
from .positionCorrection import TikePositionCorrectionController
from .probeCorrection import TikeProbeCorrectionController


class TikeParametersController:

    def __init__(self, model: TikeReconstructorLibrary, view: TikeParametersView) -> None:
        self._model = model
        self._view = view
        self._multigridController = TikeMultigridController.createInstance(
            model.multigridPresenter, view.multigridView)
        self._positionCorrectionController = TikePositionCorrectionController.createInstance(
            model.positionCorrectionPresenter, view.positionCorrectionView)
        self._probeCorrectionController = TikeProbeCorrectionController.createInstance(
            model.probeCorrectionPresenter, view.probeCorrectionView)
        self._objectCorrectionController = TikeObjectCorrectionController.createInstance(
            model.objectCorrectionPresenter, view.objectCorrectionView)
        self._basicParametersController = TikeBasicParametersController.createInstance(
            model.presenter, view.basicParametersView)

    @classmethod
    def createInstance(cls, model: TikeReconstructorLibrary,
                       view: TikeParametersView) -> TikeParametersController:
        controller = cls(model, view)
        return controller

from __future__ import annotations

from ptychodus.api.observer import Observable, Observer

from ...model.tike import TikePositionCorrectionPresenter
from ...view.tike import TikePositionCorrectionView
from .adaptiveMoment import TikeAdaptiveMomentController


class TikePositionCorrectionController(Observer):

    def __init__(
        self,
        presenter: TikePositionCorrectionPresenter,
        view: TikePositionCorrectionView,
    ) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view
        self._adaptiveMomentController = TikeAdaptiveMomentController.createInstance(
            presenter, view.adaptiveMomentView)

    @classmethod
    def createInstance(
        cls,
        presenter: TikePositionCorrectionPresenter,
        view: TikePositionCorrectionView,
    ) -> TikePositionCorrectionController:
        controller = cls(presenter, view)
        presenter.addObserver(controller)

        view.setCheckable(True)
        view.toggled.connect(presenter.setPositionCorrectionEnabled)

        view.positionRegularizationCheckBox.toggled.connect(
            presenter.setPositionRegularizationEnabled)
        view.updateMagnitudeLimitLineEdit.valueChanged.connect(presenter.setUpdateMagnitudeLimit)

        controller._syncModelToView()

        return controller

    def _syncModelToView(self) -> None:
        self._view.setChecked(self._presenter.isPositionCorrectionEnabled())

        self._view.positionRegularizationCheckBox.setChecked(
            self._presenter.isPositionRegularizationEnabled())
        self._view.updateMagnitudeLimitLineEdit.setValue(self._presenter.getUpdateMagnitudeLimit())

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()

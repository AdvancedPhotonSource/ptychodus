from __future__ import annotations

from ...api.observer import Observable, Observer
from ...model.tike import TikeObjectCorrectionPresenter
from ...view.tike import TikeObjectCorrectionView
from .adaptiveMoment import TikeAdaptiveMomentController


class TikeObjectCorrectionController(Observer):

    def __init__(self, presenter: TikeObjectCorrectionPresenter,
                 view: TikeObjectCorrectionView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view
        self._adaptiveMomentController = TikeAdaptiveMomentController.createInstance(
            presenter, view.adaptiveMomentView)

    @classmethod
    def createInstance(cls, presenter: TikeObjectCorrectionPresenter,
                       view: TikeObjectCorrectionView) -> TikeObjectCorrectionController:
        controller = cls(presenter, view)
        presenter.addObserver(controller)

        view.setCheckable(True)
        view.toggled.connect(presenter.setObjectCorrectionEnabled)
        view.positivityConstraintSlider.valueChanged.connect(presenter.setPositivityConstraint)
        view.smoothnessConstraintSlider.valueChanged.connect(presenter.setSmoothnessConstraint)
        view.useMagnitudeClippingCheckBox.toggled.connect(presenter.setMagnitudeClippingEnabled)

        controller._syncModelToView()

        return controller

    def _syncModelToView(self) -> None:
        self._view.setChecked(self._presenter.isObjectCorrectionEnabled())

        self._view.positivityConstraintSlider.setValueAndRange(
            self._presenter.getPositivityConstraint(),
            self._presenter.getPositivityConstraintLimits(),
            blockValueChangedSignal=True)
        self._view.smoothnessConstraintSlider.setValueAndRange(
            self._presenter.getSmoothnessConstraint(),
            self._presenter.getSmoothnessConstraintLimits(),
            blockValueChangedSignal=True)
        self._view.useMagnitudeClippingCheckBox.setChecked(
            self._presenter.isMagnitudeClippingEnabled())

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()

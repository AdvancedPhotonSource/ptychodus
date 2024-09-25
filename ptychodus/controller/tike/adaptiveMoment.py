from __future__ import annotations
from typing import Any

from ptychodus.api.observer import Observable, Observer

from ...model.tike import TikeAdaptiveMomentPresenter
from ...view.tike import TikeAdaptiveMomentView


class TikeAdaptiveMomentController(Observer):
    def __init__(
        self, presenter: TikeAdaptiveMomentPresenter[Any], view: TikeAdaptiveMomentView
    ) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

    @classmethod
    def createInstance(
        cls, presenter: TikeAdaptiveMomentPresenter[Any], view: TikeAdaptiveMomentView
    ) -> TikeAdaptiveMomentController:
        controller = cls(presenter, view)
        presenter.addObserver(controller)

        view.setCheckable(True)
        view.toggled.connect(presenter.setAdaptiveMomentEnabled)

        view.mdecaySlider.valueChanged.connect(presenter.setMDecay)
        view.vdecaySlider.valueChanged.connect(presenter.setVDecay)

        controller._syncModelToView()

        return controller

    def _syncModelToView(self) -> None:
        self._view.setChecked(self._presenter.isAdaptiveMomentEnabled())

        self._view.mdecaySlider.setValueAndRange(
            self._presenter.getMDecay(),
            self._presenter.getMDecayLimits(),
            blockValueChangedSignal=True,
        )
        self._view.vdecaySlider.setValueAndRange(
            self._presenter.getVDecay(),
            self._presenter.getVDecayLimits(),
            blockValueChangedSignal=True,
        )

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()

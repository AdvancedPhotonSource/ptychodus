from __future__ import annotations

from ...api.observer import Observable, Observer
from ...model.tike import TikeMultigridPresenter
from ...view.tike import TikeMultigridView


class TikeMultigridController(Observer):

    def __init__(self, presenter: TikeMultigridPresenter, view: TikeMultigridView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

    @classmethod
    def createInstance(cls, presenter: TikeMultigridPresenter,
                       view: TikeMultigridView) -> TikeMultigridController:
        controller = cls(presenter, view)
        presenter.addObserver(controller)

        view.setCheckable(True)
        view.toggled.connect(presenter.setMultigridEnabled)

        view.numLevelsSpinBox.valueChanged.connect(presenter.setNumLevels)

        controller._syncModelToView()

        return controller

    def _syncModelToView(self) -> None:
        self._view.setChecked(self._presenter.isMultigridEnabled())

        self._view.numLevelsSpinBox.blockSignals(True)
        self._view.numLevelsSpinBox.setRange(self._presenter.getNumLevelsLimits().lower,
                                             self._presenter.getNumLevelsLimits().upper)
        self._view.numLevelsSpinBox.setValue(self._presenter.getNumLevels())
        self._view.numLevelsSpinBox.blockSignals(False)

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()

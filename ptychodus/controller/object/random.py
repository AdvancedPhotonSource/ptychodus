from __future__ import annotations

from ...api.observer import Observable, Observer
from ...model.object import RandomObjectRepositoryItem
from ...view import RandomObjectView


class RandomObjectController(Observer):

    def __init__(self, item: RandomObjectRepositoryItem, view: RandomObjectView) -> None:
        super().__init__()
        self._item = item
        self._view = view

    @classmethod
    def createInstance(cls, item: RandomObjectRepositoryItem,
                       view: RandomObjectView) -> RandomObjectController:
        controller = cls(item, view)
        item.addObserver(controller)

        view.extraPaddingXSpinBox.valueChanged.connect(item.setExtraPaddingX)
        view.extraPaddingYSpinBox.valueChanged.connect(item.setExtraPaddingY)

        view.amplitudeMeanSlider.valueChanged.connect(item.setAmplitudeMean)
        view.amplitudeDeviationSlider.valueChanged.connect(item.setAmplitudeDeviation)

        view.randomizePhaseCheckBox.toggled.connect(item.setRandomizePhaseEnabled)

        controller._syncModelToView()

        return controller

    def _syncModelToView(self) -> None:
        self._view.extraPaddingXSpinBox.setValue(self._item.getExtraPaddingX())
        self._view.extraPaddingYSpinBox.setValue(self._item.getExtraPaddingY())

        self._view.amplitudeMeanSlider.setValueAndRange(self._item.getAmplitudeMean(),
                                                        self._item.getAmplitudeMeanLimits())
        self._view.amplitudeDeviationSlider.setValueAndRange(
            self._item.getAmplitudeDeviation(), self._item.getAmplitudeDeviationLimits())

        self._view.randomizePhaseCheckBox.setChecked(self._item.isRandomizePhaseEnabled())

    def update(self, observable: Observable) -> None:
        if observable is self._item:
            self._syncModelToView()

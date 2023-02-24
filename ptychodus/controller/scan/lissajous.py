from __future__ import annotations

from ...api.observer import Observable, Observer
from ...model.scan import LissajousScanRepositoryItem
from ...view import LissajousScanView


class LissajousScanController(Observer):

    def __init__(self, item: LissajousScanRepositoryItem, view: LissajousScanView) -> None:
        super().__init__()
        self._item = item
        self._view = view

    @classmethod
    def createInstance(cls, item: LissajousScanRepositoryItem,
                       view: LissajousScanView) -> LissajousScanController:
        controller = cls(item, view)
        item.addObserver(controller)

        view.numberOfPointsSpinBox.valueChanged.connect(item.setNumberOfPoints)
        view.amplitudeXWidget.lengthChanged.connect(item.setAmplitudeXInMeters)
        view.amplitudeYWidget.lengthChanged.connect(item.setAmplitudeYInMeters)
        view.angularStepXWidget.angleChanged.connect(item.setAngularStepXInTurns)
        view.angularStepYWidget.angleChanged.connect(item.setAngularStepYInTurns)
        view.angularShiftWidget.angleChanged.connect(item.setAngularShiftInTurns)

        controller._syncModelToView()

        return controller

    def _syncModelToView(self) -> None:
        self._view.numberOfPointsSpinBox.setValue(self._item.getNumberOfPoints())
        self._view.amplitudeXWidget.setLengthInMeters(self._item.getAmplitudeXInMeters())
        self._view.amplitudeYWidget.setLengthInMeters(self._item.getAmplitudeYInMeters())
        self._view.angularStepXWidget.setAngleInTurns(self._item.getAngularStepXInTurns())
        self._view.angularStepYWidget.setAngleInTurns(self._item.getAngularStepYInTurns())
        self._view.angularShiftWidget.setAngleInTurns(self._item.getAngularShiftInTurns())

    def update(self, observable: Observable) -> None:
        if observable is self._item:
            self._syncModelToView()

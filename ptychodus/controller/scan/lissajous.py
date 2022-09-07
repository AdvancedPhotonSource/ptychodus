from __future__ import annotations

from ...api.observer import Observable, Observer
from ...model import LissajousScanInitializer
from ...view import LissajousScanView


class LissajousScanController(Observer):

    def __init__(self, initializer: LissajousScanInitializer, view: LissajousScanView) -> None:
        super().__init__()
        self._initializer = initializer
        self._view = view

    @classmethod
    def createInstance(cls, initializer: LissajousScanInitializer,
                       view: LissajousScanView) -> LissajousScanController:
        controller = cls(initializer, view)
        initializer.addObserver(controller)

        view.numberOfPointsSpinBox.valueChanged.connect(initializer.setNumberOfPoints)
        view.amplitudeXWidget.lengthChanged.connect(initializer.setAmplitudeXInMeters)
        view.amplitudeYWidget.lengthChanged.connect(initializer.setAmplitudeYInMeters)
        view.angularStepXWidget.angleChanged.connect(initializer.setAngularStepXInTurns)
        view.angularStepYWidget.angleChanged.connect(initializer.setAngularStepYInTurns)
        view.angularShiftWidget.angleChanged.connect(initializer.setAngularShiftInTurns)

        controller._syncModelToView()

        return controller

    def _syncModelToView(self) -> None:
        self._view.numberOfPointsSpinBox.setValue(self._initializer.getNumberOfPoints())
        self._view.amplitudeXWidget.setLengthInMeters(self._initializer.getAmplitudeXInMeters())
        self._view.amplitudeYWidget.setLengthInMeters(self._initializer.getAmplitudeYInMeters())
        self._view.angularStepXWidget.setAngleInTurns(self._initializer.getAngularStepXInTurns())
        self._view.angularStepYWidget.setAngleInTurns(self._initializer.getAngularStepYInTurns())
        self._view.angularShiftWidget.setAngleInTurns(self._initializer.getAngularShiftInTurns())

    def update(self, observable: Observable) -> None:
        if observable is self._initializer:
            self._syncModelToView()

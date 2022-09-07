from __future__ import annotations

from ...api.observer import Observable, Observer
from ...model import SpiralScanInitializer
from ...view import SpiralScanView


class SpiralScanController(Observer):

    def __init__(self, initializer: SpiralScanInitializer, view: SpiralScanView) -> None:
        super().__init__()
        self._initializer = initializer
        self._view = view

    @classmethod
    def createInstance(cls, initializer: SpiralScanInitializer,
                       view: SpiralScanView) -> SpiralScanController:
        controller = cls(initializer, view)
        initializer.addObserver(controller)

        view.numberOfPointsSpinBox.valueChanged.connect(initializer.setNumberOfPoints)
        view.radiusScalarWidget.lengthChanged.connect(initializer.setRadiusScalarInMeters)
        view.angularStepWidget.angleChanged.connect(initializer.setAngularStepInTurns)

        controller._syncModelToView()

        return controller

    def _syncModelToView(self) -> None:
        self._view.numberOfPointsSpinBox.setValue(self._initializer.getNumberOfPoints())
        self._view.radiusScalarWidget.setLengthInMeters(
            self._initializer.getRadiusScalarInMeters())
        self._view.angularStepWidget.setAngleInTurns(self._initializer.getAngularStepInTurns())

    def update(self, observable: Observable) -> None:
        if observable is self._initializer:
            self._syncModelToView()

from __future__ import annotations

from ...api.observer import Observable, Observer
from ...model import CartesianScanInitializer
from ...view import CartesianScanView


class CartesianScanController(Observer):

    def __init__(self, initializer: CartesianScanInitializer, view: CartesianScanView) -> None:
        super().__init__()
        self._initializer = initializer
        self._view = view

    @classmethod
    def createInstance(cls, initializer: CartesianScanInitializer,
                       view: CartesianScanView) -> CartesianScanController:
        controller = cls(initializer, view)
        initializer.addObserver(controller)

        view.numberOfPointsXSpinBox.valueChanged.connect(initializer.setNumberOfPointsX)
        view.numberOfPointsYSpinBox.valueChanged.connect(initializer.setNumberOfPointsY)

        view.stepSizeXWidget.lengthChanged.connect(initializer.setStepSizeXInMeters)
        view.stepSizeYWidget.lengthChanged.connect(initializer.setStepSizeYInMeters)

        controller._syncModelToView()

        return controller

    def _syncModelToView(self) -> None:
        self._view.numberOfPointsXSpinBox.setValue(self._initializer.getNumberOfPointsX())
        self._view.numberOfPointsYSpinBox.setValue(self._initializer.getNumberOfPointsY())

        self._view.stepSizeXWidget.setLengthInMeters(self._initializer.getStepSizeXInMeters())
        self._view.stepSizeYWidget.setLengthInMeters(self._initializer.getStepSizeYInMeters())

    def update(self, observable: Observable) -> None:
        if observable is self._initializer:
            self._syncModelToView()

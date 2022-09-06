from __future__ import annotations

from ...api.observer import Observable, Observer
from ...model import SpiralScanInitializer
from ...view import ScanEditorView


class SpiralScanController(Observer):

    def __init__(self, initializer: SpiralScanInitializer, view: ScanEditorView) -> None:
        super().__init__()
        self._initializer = initializer
        self._view = view

    @classmethod
    def createInstance(cls, initializer: SpiralScanInitializer,
                       view: ScanEditorView) -> SpiralScanController:
        controller = cls(initializer, view)
        initializer.addObserver(controller)

        view.numberOfPointsSpinBox.valueChanged.connect(initializer.setNumberOfPoints)
        view.radiusScalarWidget.lengthChanged.connect(initializer.setRadiusScalarInMeters)
        view.stepSizeWidget.lengthChanged.connect(initializer.setStepSizeInTurns)

        controller._syncModelToView()

        return controller

    def _syncModelToView(self) -> None:
        self._view.numberOfPointsSpinBox.setValue(self._initializer.getNumberOfPoints())
        self._view.radiusScalarWidget.setLengthInMeters(self._initializer.getRadiusScalarInMeters())
        self._view.stepSizeWidget.setLengthInMeters(self._initializer.getStepSizeInTurns())

    def update(self, observable: Observable) -> None:
        if observable is self._initializer:
            self._syncModelToView()

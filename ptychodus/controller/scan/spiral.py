from __future__ import annotations

from ...api.observer import Observable, Observer
from ...model.scan import SpiralScanRepositoryItem
from ...view import SpiralScanView


class SpiralScanController(Observer):

    def __init__(self, item: SpiralScanRepositoryItem, view: SpiralScanView) -> None:
        super().__init__()
        self._item = item
        self._view = view

    @classmethod
    def createInstance(cls, item: SpiralScanRepositoryItem,
                       view: SpiralScanView) -> SpiralScanController:
        controller = cls(item, view)
        item.addObserver(controller)

        view.numberOfPointsSpinBox.valueChanged.connect(item.setNumberOfPoints)
        view.radiusScalarWidget.lengthChanged.connect(item.setRadiusScalarInMeters)
        view.angularStepWidget.angleChanged.connect(item.setAngularStepInTurns)

        controller._syncModelToView()

        return controller

    def _syncModelToView(self) -> None:
        self._view.numberOfPointsSpinBox.setValue(self._item.getNumberOfPoints())
        self._view.radiusScalarWidget.setLengthInMeters(self._item.getRadiusScalarInMeters())
        self._view.angularStepWidget.setAngleInTurns(self._item.getAngularStepInTurns())

    def update(self, observable: Observable) -> None:
        if observable is self._item:
            self._syncModelToView()

from __future__ import annotations

from ...api.observer import Observable, Observer
from ...model import CartesianScanRepositoryItem
from ...view import CartesianScanView


class CartesianScanController(Observer):

    def __init__(self, item: CartesianScanRepositoryItem, view: CartesianScanView) -> None:
        super().__init__()
        self._item = item
        self._view = view

    @classmethod
    def createInstance(cls, item: CartesianScanRepositoryItem,
                       view: CartesianScanView) -> CartesianScanController:
        controller = cls(item, view)
        item.addObserver(controller)

        view.numberOfPointsXSpinBox.valueChanged.connect(item.setNumberOfPointsX)
        view.numberOfPointsYSpinBox.valueChanged.connect(item.setNumberOfPointsY)

        view.stepSizeXWidget.lengthChanged.connect(item.setStepSizeXInMeters)
        view.stepSizeYWidget.lengthChanged.connect(item.setStepSizeYInMeters)

        controller._syncModelToView()

        return controller

    def _syncModelToView(self) -> None:
        self._view.numberOfPointsXSpinBox.setValue(self._item.getNumberOfPointsX())
        self._view.numberOfPointsYSpinBox.setValue(self._item.getNumberOfPointsY())

        self._view.stepSizeXWidget.setLengthInMeters(self._item.getStepSizeXInMeters())
        self._view.stepSizeYWidget.setLengthInMeters(self._item.getStepSizeYInMeters())

    def update(self, observable: Observable) -> None:
        if observable is self._item:
            self._syncModelToView()

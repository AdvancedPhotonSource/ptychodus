from __future__ import annotations

from ...api.observer import Observable, Observer
from ...model.scan import TransformedScanRepositoryItem
from ...view import ScanTransformView


class ScanTransformController(Observer):

    def __init__(self, item: TransformedScanRepositoryItem, view: ScanTransformView) -> None:
        super().__init__()
        self._item = item
        self._view = view

    @classmethod
    def createInstance(cls, item: TransformedScanRepositoryItem,
                       view: ScanTransformView) -> ScanTransformController:
        controller = cls(item, view)
        item.addObserver(controller)

        for name in item.getTransformNameList():
            view.transformComboBox.addItem(name)

        view.transformComboBox.currentTextChanged.connect(item.setTransformByName)
        view.jitterRadiusWidget.lengthChanged.connect(item.setJitterRadiusInMeters)
        view.centroidXWidget.lengthChanged.connect(item.setCentroidXInMeters)
        view.centroidYWidget.lengthChanged.connect(item.setCentroidYInMeters)

        controller._syncModelToView()

        return controller

    def _syncModelToView(self) -> None:
        self._view.transformComboBox.setCurrentText(self._item.getTransformName())
        self._view.jitterRadiusWidget.setLengthInMeters(self._item.getJitterRadiusInMeters())
        self._view.centroidXWidget.setLengthInMeters(self._item.getCentroidXInMeters())
        self._view.centroidYWidget.setLengthInMeters(self._item.getCentroidYInMeters())

    def update(self, observable: Observable) -> None:
        if observable is self._item:
            self._syncModelToView()

from __future__ import annotations

from ...api.observer import Observable, Observer
from ...model.scan import ScanRepositoryItem
from ...view.scan import ScanTransformView


class ScanTransformController(Observer):

    def __init__(self, item: ScanRepositoryItem, view: ScanTransformView) -> None:
        super().__init__()
        self._item = item
        self._view = view

    @classmethod
    def createInstance(cls, item: ScanRepositoryItem,
                       view: ScanTransformView) -> ScanTransformController:
        controller = cls(item, view)
        item.addObserver(controller)

        for name in item.getIndexFilterNameList():
            view.indexFilterComboBox.addItem(name)

        view.indexFilterComboBox.textActivated.connect(item.setIndexFilterByName)

        for name in item.getTransformNameList():
            view.transformComboBox.addItem(name)

        view.transformComboBox.textActivated.connect(item.setTransformByName)
        view.jitterRadiusWidget.lengthChanged.connect(item.setJitterRadiusInMeters)

        view.centroidXCheckBox.toggled.connect(item.setOverrideCentroidXEnabled)
        view.centroidXWidget.lengthChanged.connect(item.setCentroidXInMeters)

        view.centroidYCheckBox.toggled.connect(item.setOverrideCentroidYEnabled)
        view.centroidYWidget.lengthChanged.connect(item.setCentroidYInMeters)

        controller._syncModelToView()

        return controller

    def _syncModelToView(self) -> None:
        self._view.indexFilterComboBox.setCurrentText(self._item.getIndexFilterName())
        self._view.transformComboBox.setCurrentText(self._item.getTransformName())
        self._view.jitterRadiusWidget.setLengthInMeters(self._item.getJitterRadiusInMeters())

        self._view.centroidXCheckBox.setChecked(self._item.isOverrideCentroidXEnabled)
        self._view.centroidXWidget.setEnabled(self._item.isOverrideCentroidXEnabled)
        self._view.centroidXWidget.setLengthInMeters(self._item.getCentroidXInMeters())

        self._view.centroidYCheckBox.setChecked(self._item.isOverrideCentroidYEnabled)
        self._view.centroidYWidget.setEnabled(self._item.isOverrideCentroidYEnabled)
        self._view.centroidYWidget.setLengthInMeters(self._item.getCentroidYInMeters())

    def update(self, observable: Observable) -> None:
        if observable is self._item:
            self._syncModelToView()

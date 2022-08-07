from __future__ import annotations

from ...api.observer import Observable, Observer
from ...model import ScanInitializer
from ...view import ScanTransformView


class ScanTransformController(Observer):

    def __init__(self, initializer: ScanInitializer, view: ScanTransformView) -> None:
        super().__init__()
        self._initializer = initializer
        self._view = view

    @classmethod
    def createInstance(cls, initializer: ScanInitializer,
                       view: ScanTransformView) -> ScanTransformController:
        controller = cls(initializer, view)
        initializer.addObserver(controller)

        for name in initializer.getTransformNameList():
            view.transformComboBox.addItem(name)

        view.transformComboBox.currentTextChanged.connect(initializer.setTransformByName)
        view.jitterRadiusWidget.lengthChanged.connect(initializer.setJitterRadiusInMeters)

        controller._syncModelToView()

        return controller

    def _syncModelToView(self) -> None:
        self._view.transformComboBox.setCurrentText(self._initializer.getTransformName())
        self._view.jitterRadiusWidget.setLengthInMeters(
            self._initializer.getJitterRadiusInMeters())

    def update(self, observable: Observable) -> None:
        if observable is self._initializer:
            self._syncModelToView()

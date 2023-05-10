from __future__ import annotations

from ...api.observer import Observable, Observer
from ...model.probe import SuperGaussianProbeInitializer
from ...view.probe import SuperGaussianProbeView


class SuperGaussianProbeController(Observer):

    def __init__(self, initializer: SuperGaussianProbeInitializer,
                 view: SuperGaussianProbeView) -> None:
        super().__init__()
        self._initializer = initializer
        self._view = view

    @classmethod
    def createInstance(cls, initializer: SuperGaussianProbeInitializer,
                       view: SuperGaussianProbeView) -> SuperGaussianProbeController:
        controller = cls(initializer, view)
        initializer.addObserver(controller)

        view.annularRadiusWidget.lengthChanged.connect(initializer.setAnnularRadiusInMeters)
        view.probeWidthWidget.lengthChanged.connect(initializer.setProbeWidthInMeters)
        view.orderParameterWidget.valueChanged.connect(initializer.setOrderParameter)
        view.numberOfModesSpinBox.valueChanged.connect(initializer.setNumberOfProbeModes)

        controller._syncModelToView()

        return controller

    def _syncModelToView(self) -> None:
        self._view.annularRadiusWidget.setLengthInMeters(
            self._initializer.getAnnularRadiusInMeters())
        self._view.probeWidthWidget.setLengthInMeters(self._initializer.getProbeWidthInMeters())
        self._view.orderParameterWidget.setValue(self._initializer.getOrderParameter())
        self._view.numberOfModesSpinBox.setValue(self._initializer.getNumberOfProbeModes())

    def update(self, observable: Observable) -> None:
        if observable is self._initializer:
            self._syncModelToView()

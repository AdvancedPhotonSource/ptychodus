from __future__ import annotations

from ...api.observer import Observable, Observer
from ...model.probe import FresnelZonePlateProbeInitializer
from ...view.probe import FresnelZonePlateProbeView


class FresnelZonePlateProbeController(Observer):

    def __init__(self, initializer: FresnelZonePlateProbeInitializer,
                 view: FresnelZonePlateProbeView) -> None:
        super().__init__()
        self._initializer = initializer
        self._view = view

    @classmethod
    def createInstance(cls, initializer: FresnelZonePlateProbeInitializer,
                       view: FresnelZonePlateProbeView) -> FresnelZonePlateProbeController:
        controller = cls(initializer, view)
        initializer.addObserver(controller)

        view.zonePlateRadiusWidget.lengthChanged.connect(initializer.setZonePlateRadiusInMeters)
        view.outermostZoneWidthWidget.lengthChanged.connect(
            initializer.setOutermostZoneWidthInMeters)
        view.beamstopDiameterWidget.lengthChanged.connect(
            initializer.setCentralBeamstopDiameterInMeters)
        view.defocusDistanceWidget.lengthChanged.connect(initializer.setDefocusDistanceInMeters)
        view.numberOfModesSpinBox.valueChanged.connect(initializer.setNumberOfProbeModes)

        controller._syncModelToView()

        return controller

    def _syncModelToView(self) -> None:
        self._view.zonePlateRadiusWidget.setLengthInMeters(
            self._initializer.getZonePlateRadiusInMeters())
        self._view.outermostZoneWidthWidget.setLengthInMeters(
            self._initializer.getOutermostZoneWidthInMeters())
        self._view.beamstopDiameterWidget.setLengthInMeters(
            self._initializer.getCentralBeamstopDiameterInMeters())
        self._view.defocusDistanceWidget.setLengthInMeters(
            self._initializer.getDefocusDistanceInMeters())
        self._view.numberOfModesSpinBox.setValue(self._initializer.getNumberOfProbeModes())

    def update(self, observable: Observable) -> None:
        if observable is self._initializer:
            self._syncModelToView()

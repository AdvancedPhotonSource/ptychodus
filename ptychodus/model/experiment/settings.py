from __future__ import annotations

from ...api.observer import Observable, Observer
from ...api.settings import SettingsRegistry, SettingsGroup


class ExperimentSettings(Observable, Observer):  # FIXME do we need this?

    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup
        self.detectorObjectDistanceInMeters = settingsGroup.createRealEntry(
            'DetectorObjectDistanceInMeters', '2')
        self.probeEnergyInElectronVolts = settingsGroup.createRealEntry(
            'ProbeEnergyInElectronVolts', '10000')
        # vvv FIXME USE IN SIZER vvv
        self.expandScanBoundingBox = settingsGroup.createBooleanEntry(
            'ExpandScanBoundingBox', False)
        self.scanBoundingBoxMinimumXInMeters = settingsGroup.createRealEntry(
            'ScanBoundingBoxMinimumXInMeters', '0')
        self.scanBoundingBoxMaximumXInMeters = settingsGroup.createRealEntry(
            'ScanBoundingBoxMaximumXInMeters', '1e-5')
        self.scanBoundingBoxMinimumYInMeters = settingsGroup.createRealEntry(
            'ScanBoundingBoxMinimumYInMeters', '0')
        self.scanBoundingBoxMaximumYInMeters = settingsGroup.createRealEntry(
            'ScanBoundingBoxMaximumYInMeters', '1e-5')
        # ^^^ FIXME USE IN SIZER ^^^

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> ExperimentSettings:
        settingsGroup = settingsRegistry.createGroup('Experiment')
        settings = cls(settingsGroup)
        settingsGroup.addObserver(settings)
        return settings

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()

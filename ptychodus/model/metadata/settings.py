from __future__ import annotations

from ...api.observer import Observable, Observer
from ...api.settings import SettingsGroup, SettingsRegistry


class MetadataSettings(Observable, Observer):

    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup
        self.probeEnergyInElectronVolts = settingsGroup.createRealEntry(
            'ProbeEnergyInElectronVolts', '10000')
        self.detectorDistanceInMeters = settingsGroup.createRealEntry(
            'DetectorDistanceInMeters', '1')

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> MetadataSettings:
        settingsGroup = settingsRegistry.createGroup('Metadata')
        settings = cls(settingsGroup)
        settingsGroup.addObserver(settings)
        return settings

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()

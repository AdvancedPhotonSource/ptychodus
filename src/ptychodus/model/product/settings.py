from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class ProductSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.createGroup('Products')
        self._settingsGroup.addObserver(self)

        self.name = self._settingsGroup.createStringParameter('Name', 'Unnamed')
        self.fileType = self._settingsGroup.createStringParameter('FileType', 'HDF5')
        self.detectorDistanceInMeters = self._settingsGroup.createRealParameter(
            'DetectorDistanceInMeters', 1.0, minimum=0.0
        )
        self.probeEnergyInElectronVolts = self._settingsGroup.createRealParameter(
            'ProbeEnergyInElectronVolts', 10000.0, minimum=0.0
        )
        self.probePhotonCount = self._settingsGroup.createRealParameter(
            'ProbePhotonCount', 0.0, minimum=0.0
        )
        self.exposureTimeInSeconds = self._settingsGroup.createRealParameter(
            'ExposureTimeInSeconds', 0.0, minimum=0.0
        )

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()

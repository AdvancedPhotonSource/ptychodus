from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class ProductSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.create_group('Products')
        self._settingsGroup.add_observer(self)

        self.name = self._settingsGroup.create_string_parameter('Name', 'Unnamed')
        self.fileType = self._settingsGroup.create_string_parameter('FileType', 'HDF5')
        self.detectorDistanceInMeters = self._settingsGroup.create_real_parameter(
            'DetectorDistanceInMeters', 1.0, minimum=0.0
        )
        self.probeEnergyInElectronVolts = self._settingsGroup.create_real_parameter(
            'ProbeEnergyInElectronVolts', 10000.0, minimum=0.0
        )
        self.probePhotonCount = self._settingsGroup.create_real_parameter(
            'ProbePhotonCount', 0.0, minimum=0.0
        )
        self.exposureTimeInSeconds = self._settingsGroup.create_real_parameter(
            'ExposureTimeInSeconds', 0.0, minimum=0.0
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notify_observers()

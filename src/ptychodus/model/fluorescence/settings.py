from pathlib import Path

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class FluorescenceSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.create_group('Fluorescence')
        self._settingsGroup.add_observer(self)

        self.filePath = self._settingsGroup.create_path_parameter(
            'FilePath', Path('/path/to/dataset.h5')
        )
        self.fileType = self._settingsGroup.create_string_parameter('FileType', 'XRF-Maps')
        self.algorithm = self._settingsGroup.create_string_parameter('Algorithm', 'VSPI')
        self.vspiDampingFactor = self._settingsGroup.create_real_parameter(
            'VSPIDampingFactor', 0.0, minimum=0.0
        )
        self.vspiMaxIterations = self._settingsGroup.create_integer_parameter(
            'VSPIMaxIterations', 100, minimum=1
        )
        self.upscalingStrategy = self._settingsGroup.create_string_parameter(
            'UpscalingStrategy', 'Linear'
        )
        self.deconvolutionStrategy = self._settingsGroup.create_string_parameter(
            'DeconvolutionStrategy', 'Richardson-Lucy'
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notify_observers()

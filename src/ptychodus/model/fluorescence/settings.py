from pathlib import Path

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class FluorescenceSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settings_group = registry.create_group('Fluorescence')
        self._settings_group.add_observer(self)

        self.file_path = self._settings_group.create_path_parameter(
            'FilePath', Path('/path/to/dataset.h5')
        )
        self.file_type = self._settings_group.create_string_parameter('FileType', 'XRF-Maps')
        self.algorithm = self._settings_group.create_string_parameter('Algorithm', 'VSPI')
        self.vspi_damping_factor = self._settings_group.create_real_parameter(
            'VSPIDampingFactor', 0.0, minimum=0.0
        )
        self.vspi_max_iterations = self._settings_group.create_integer_parameter(
            'VSPIMaxIterations', 100, minimum=1
        )
        self.upscaling_strategy = self._settings_group.create_string_parameter(
            'UpscalingStrategy', 'Linear'
        )
        self.deconvolution_strategy = self._settings_group.create_string_parameter(
            'DeconvolutionStrategy', 'Richardson-Lucy'
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._settings_group:
            self.notify_observers()

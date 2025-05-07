from pathlib import Path

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class FluorescenceSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._group = registry.create_group('Fluorescence')
        self._group.add_observer(self)

        self.file_path = self._group.create_path_parameter('FilePath', Path('/path/to/dataset.h5'))
        self.file_type = self._group.create_string_parameter('FileType', 'XRF-Maps')
        self.algorithm = self._group.create_string_parameter('Algorithm', 'VSPI')
        self.vspi_damping_factor = self._group.create_real_parameter(
            'VSPIDampingFactor', 0.0, minimum=0.0
        )
        self.vspi_max_iterations = self._group.create_integer_parameter(
            'VSPIMaxIterations', 100, minimum=1
        )
        self.upscaling_strategy = self._group.create_string_parameter('UpscalingStrategy', 'Linear')
        self.deconvolution_strategy = self._group.create_string_parameter(
            'DeconvolutionStrategy', 'Richardson-Lucy'
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()

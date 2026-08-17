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
            'DeconvolutionStrategy', 'RichardsonLucy'
        )

        # Ptychozoon (GPU VSPI) settings
        self.ptychozoon_damping_factor = self._group.create_real_parameter(
            'PtychozoonDampingFactor', 0.0, minimum=0.0
        )
        self.ptychozoon_gradient_smoothness = self._group.create_real_parameter(
            'PtychozoonGradientSmoothness', 0.0, minimum=0.0
        )
        self.ptychozoon_max_iterations = self._group.create_integer_parameter(
            'PtychozoonMaxIterations', 100, minimum=1
        )
        self.ptychozoon_atol = self._group.create_real_parameter(
            'PtychozoonATol', 1e-6, minimum=0.0
        )
        self.ptychozoon_btol = self._group.create_real_parameter(
            'PtychozoonBTol', 1e-6, minimum=0.0
        )
        self.ptychozoon_checkpoint_interval = self._group.create_integer_parameter(
            'PtychozoonCheckpointInterval', 5, minimum=1
        )
        self.ptychozoon_use_gpu = self._group.create_boolean_parameter('PtychozoonUseGPU', True)
        self.ptychozoon_gpu_device_index = self._group.create_integer_parameter(
            'PtychozoonGPUDeviceIndex', 0, minimum=0
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()

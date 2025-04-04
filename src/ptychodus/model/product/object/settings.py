from pathlib import Path

import numpy

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class ObjectSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settings_group = registry.create_group('Object')
        self._settings_group.add_observer(self)

        self.builder = self._settings_group.create_string_parameter('Builder', 'Random')
        self.file_path = self._settings_group.create_path_parameter(
            'FilePath', Path('/path/to/object.npy')
        )
        self.file_type = self._settings_group.create_string_parameter('FileType', 'NPY')

        self.object_layer_distance_m = self._settings_group.create_real_sequence_parameter(
            'ObjectLayerDistanceInMeters', []
        )

        self.extra_padding_x = self._settings_group.create_integer_parameter(
            'ExtraPaddingX', 1, minimum=0
        )
        self.extra_padding_y = self._settings_group.create_integer_parameter(
            'ExtraPaddingY', 1, minimum=0
        )
        self.amplitude_mean = self._settings_group.create_real_parameter(
            'AmplitudeMean', 1.0, minimum=0.0, maximum=1.0
        )
        self.amplitude_deviation = self._settings_group.create_real_parameter(
            'AmplitudeDeviation', 0.0, minimum=0.0, maximum=1.0
        )
        self.phase_deviation = self._settings_group.create_real_parameter(
            'PhaseDeviation', 0.0, minimum=0.0, maximum=numpy.pi
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._settings_group:
            self.notify_observers()

from pathlib import Path

import numpy

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class ObjectSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.create_group('Object')
        self._settingsGroup.add_observer(self)

        self.builder = self._settingsGroup.create_string_parameter('Builder', 'Random')
        self.filePath = self._settingsGroup.create_path_parameter(
            'FilePath', Path('/path/to/object.npy')
        )
        self.fileType = self._settingsGroup.create_string_parameter('FileType', 'NPY')

        self.objectLayerDistanceInMeters = self._settingsGroup.create_real_sequence_parameter(
            'ObjectLayerDistanceInMeters', []
        )

        self.extraPaddingX = self._settingsGroup.create_integer_parameter(
            'ExtraPaddingX', 1, minimum=0
        )
        self.extraPaddingY = self._settingsGroup.create_integer_parameter(
            'ExtraPaddingY', 1, minimum=0
        )
        self.amplitudeMean = self._settingsGroup.create_real_parameter(
            'AmplitudeMean', 1.0, minimum=0.0, maximum=1.0
        )
        self.amplitudeDeviation = self._settingsGroup.create_real_parameter(
            'AmplitudeDeviation', 0.0, minimum=0.0, maximum=1.0
        )
        self.phaseDeviation = self._settingsGroup.create_real_parameter(
            'PhaseDeviation', 0.0, minimum=0.0, maximum=numpy.pi
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notify_observers()

from pathlib import Path

import numpy

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class ObjectSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.createGroup('Object')
        self._settingsGroup.addObserver(self)

        self.builder = self._settingsGroup.createStringParameter('Builder', 'Random')
        self.filePath = self._settingsGroup.createPathParameter(
            'FilePath', Path('/path/to/object.npy')
        )
        self.fileType = self._settingsGroup.createStringParameter('FileType', 'NPY')

        self.objectLayerDistanceInMeters = self._settingsGroup.createRealSequenceParameter(
            'ObjectLayerDistanceInMeters', []
        )

        self.extraPaddingX = self._settingsGroup.createIntegerParameter(
            'ExtraPaddingX', 1, minimum=0
        )
        self.extraPaddingY = self._settingsGroup.createIntegerParameter(
            'ExtraPaddingY', 1, minimum=0
        )
        self.amplitudeMean = self._settingsGroup.createRealParameter(
            'AmplitudeMean', 1.0, minimum=0.0, maximum=1.0
        )
        self.amplitudeDeviation = self._settingsGroup.createRealParameter(
            'AmplitudeDeviation', 0.0, minimum=0.0, maximum=1.0
        )
        self.phaseDeviation = self._settingsGroup.createRealParameter(
            'PhaseDeviation', 0.0, minimum=0.0, maximum=numpy.pi
        )

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()

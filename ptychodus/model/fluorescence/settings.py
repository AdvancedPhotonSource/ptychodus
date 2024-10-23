from pathlib import Path

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class FluorescenceSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.createGroup('Fluorescence')
        self._settingsGroup.addObserver(self)

        self.filePath = self._settingsGroup.createPathParameter(
            'FilePath', Path('/path/to/dataset.h5')
        )
        self.fileType = self._settingsGroup.createStringParameter('FileType', 'XRF-Maps')
        self.algorithm = self._settingsGroup.createStringParameter('Algorithm', 'VSPI')
        self.vspiDampingFactor = self._settingsGroup.createRealParameter(
            'VSPIDampingFactor', 0.0, minimum=0.0
        )
        self.vspiMaxIterations = self._settingsGroup.createIntegerParameter(
            'VSPIMaxIterations', 100, minimum=1
        )
        self.upscalingStrategy = self._settingsGroup.createStringParameter(
            'UpscalingStrategy', 'Linear'
        )
        self.deconvolutionStrategy = self._settingsGroup.createStringParameter(
            'DeconvolutionStrategy', 'Richardson-Lucy'
        )

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()

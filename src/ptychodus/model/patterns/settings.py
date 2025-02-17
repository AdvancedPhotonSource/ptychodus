from pathlib import Path

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class DetectorSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.createGroup('Detector')
        self._settingsGroup.addObserver(self)

        self.widthInPixels = self._settingsGroup.createIntegerParameter(
            'WidthInPixels', 1024, minimum=1
        )
        self.pixelWidthInMeters = self._settingsGroup.createRealParameter(
            'PixelWidthInMeters', 75e-6, minimum=0.0
        )
        self.heightInPixels = self._settingsGroup.createIntegerParameter(
            'HeightInPixels', 1024, minimum=1
        )
        self.pixelHeightInMeters = self._settingsGroup.createRealParameter(
            'PixelHeightInMeters', 75e-6, minimum=0.0
        )
        self.bitDepth = self._settingsGroup.createIntegerParameter('BitDepth', 8, minimum=1)

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()


class PatternSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.createGroup('Patterns')
        self._settingsGroup.addObserver(self)

        self.fileType = self._settingsGroup.createStringParameter('FileType', 'NeXus')
        self.filePath = self._settingsGroup.createPathParameter(
            'FilePath', Path('/path/to/data.h5')
        )
        self.memmapEnabled = self._settingsGroup.createBooleanParameter('MemmapEnabled', False)
        self.scratchDirectory = self._settingsGroup.createPathParameter(
            'ScratchDirectory', Path.home() / '.ptychodus'
        )
        self.numberOfDataThreads = self._settingsGroup.createIntegerParameter(
            'NumberOfDataThreads', 8, minimum=1, maximum=64
        )

        self.cropEnabled = self._settingsGroup.createBooleanParameter('CropEnabled', True)
        self.cropCenterXInPixels = self._settingsGroup.createIntegerParameter(
            'CropCenterXInPixels', 32, minimum=0
        )
        self.cropCenterYInPixels = self._settingsGroup.createIntegerParameter(
            'CropCenterYInPixels', 32, minimum=0
        )
        self.cropWidthInPixels = self._settingsGroup.createIntegerParameter(
            'CropWidthInPixels', 64, minimum=1
        )
        self.cropHeightInPixels = self._settingsGroup.createIntegerParameter(
            'CropHeightInPixels', 64, minimum=1
        )

        self.binningEnabled = self._settingsGroup.createBooleanParameter('BinningEnabled', False)
        self.binSizeX = self._settingsGroup.createIntegerParameter('BinSizeX', 1, minimum=1)
        self.binSizeY = self._settingsGroup.createIntegerParameter('BinSizeY', 1, minimum=1)

        self.paddingEnabled = self._settingsGroup.createBooleanParameter('PaddingEnabled', False)
        self.padX = self._settingsGroup.createIntegerParameter('PadX', 0, minimum=0)
        self.padY = self._settingsGroup.createIntegerParameter('PadY', 0, minimum=0)

        self.flipXEnabled = self._settingsGroup.createBooleanParameter('FlipXEnabled', False)
        self.flipYEnabled = self._settingsGroup.createBooleanParameter('FlipYEnabled', False)

        self.valueLowerBoundEnabled = self._settingsGroup.createBooleanParameter(
            'ValueLowerBoundEnabled', False
        )
        self.valueLowerBound = self._settingsGroup.createIntegerParameter(
            'ValueLowerBound', 0, minimum=0
        )
        self.valueUpperBoundEnabled = self._settingsGroup.createBooleanParameter(
            'ValueUpperBoundEnabled', False
        )
        self.valueUpperBound = self._settingsGroup.createIntegerParameter(
            'ValueUpperBound', 65535, minimum=0
        )

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()

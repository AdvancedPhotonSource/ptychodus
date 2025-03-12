from pathlib import Path

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class DetectorSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.create_group('Detector')
        self._settingsGroup.add_observer(self)

        self.widthInPixels = self._settingsGroup.create_integer_parameter(
            'WidthInPixels', 1024, minimum=1
        )
        self.pixelWidthInMeters = self._settingsGroup.create_real_parameter(
            'PixelWidthInMeters', 75e-6, minimum=0.0
        )
        self.heightInPixels = self._settingsGroup.create_integer_parameter(
            'HeightInPixels', 1024, minimum=1
        )
        self.pixelHeightInMeters = self._settingsGroup.create_real_parameter(
            'PixelHeightInMeters', 75e-6, minimum=0.0
        )
        self.bitDepth = self._settingsGroup.create_integer_parameter('BitDepth', 8, minimum=1)

    def _update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notify_observers()


class PatternSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.create_group('Patterns')
        self._settingsGroup.add_observer(self)

        self.fileType = self._settingsGroup.create_string_parameter('FileType', 'NeXus')
        self.filePath = self._settingsGroup.create_path_parameter(
            'FilePath', Path('/path/to/data.h5')
        )
        self.memmapEnabled = self._settingsGroup.create_boolean_parameter('MemmapEnabled', False)
        self.scratchDirectory = self._settingsGroup.create_path_parameter(
            'ScratchDirectory', Path.home() / '.ptychodus'
        )
        self.numberOfDataThreads = self._settingsGroup.create_integer_parameter(
            'NumberOfDataThreads', 8, minimum=1, maximum=64
        )

        self.cropEnabled = self._settingsGroup.create_boolean_parameter('CropEnabled', True)
        self.cropCenterXInPixels = self._settingsGroup.create_integer_parameter(
            'CropCenterXInPixels', 32, minimum=0
        )
        self.cropCenterYInPixels = self._settingsGroup.create_integer_parameter(
            'CropCenterYInPixels', 32, minimum=0
        )
        self.cropWidthInPixels = self._settingsGroup.create_integer_parameter(
            'CropWidthInPixels', 64, minimum=1
        )
        self.cropHeightInPixels = self._settingsGroup.create_integer_parameter(
            'CropHeightInPixels', 64, minimum=1
        )

        self.binningEnabled = self._settingsGroup.create_boolean_parameter('BinningEnabled', False)
        self.binSizeX = self._settingsGroup.create_integer_parameter('BinSizeX', 1, minimum=1)
        self.binSizeY = self._settingsGroup.create_integer_parameter('BinSizeY', 1, minimum=1)

        self.paddingEnabled = self._settingsGroup.create_boolean_parameter('PaddingEnabled', False)
        self.padX = self._settingsGroup.create_integer_parameter('PadX', 0, minimum=0)
        self.padY = self._settingsGroup.create_integer_parameter('PadY', 0, minimum=0)

        self.flipXEnabled = self._settingsGroup.create_boolean_parameter('FlipXEnabled', False)
        self.flipYEnabled = self._settingsGroup.create_boolean_parameter('FlipYEnabled', False)

        self.valueLowerBoundEnabled = self._settingsGroup.create_boolean_parameter(
            'ValueLowerBoundEnabled', False
        )
        self.valueLowerBound = self._settingsGroup.create_integer_parameter(
            'ValueLowerBound', 0, minimum=0
        )
        self.valueUpperBoundEnabled = self._settingsGroup.create_boolean_parameter(
            'ValueUpperBoundEnabled', False
        )
        self.valueUpperBound = self._settingsGroup.create_integer_parameter(
            'ValueUpperBound', 65535, minimum=0
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notify_observers()

from __future__ import annotations

from ptychodus.api.geometry import ImageExtent, PixelGeometry
from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class Detector(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.createGroup('Detector')
        self._settingsGroup.addObserver(self)

        self.widthInPixels = self._settingsGroup.createIntegerParameter(
            'WidthInPixels', 1024, minimum=0
        )
        self.pixelWidthInMeters = self._settingsGroup.createRealParameter(
            'PixelWidthInMeters', 75e-6, minimum=0.0
        )
        self.heightInPixels = self._settingsGroup.createIntegerParameter(
            'HeightInPixels', 1024, minimum=0
        )
        self.pixelHeightInMeters = self._settingsGroup.createRealParameter(
            'PixelHeightInMeters', 75e-6, minimum=0.0
        )
        self.bitDepth = self._settingsGroup.createIntegerParameter('BitDepth', 8, minimum=1)

    def getImageExtent(self) -> ImageExtent:
        return ImageExtent(
            widthInPixels=self.widthInPixels.getValue(),
            heightInPixels=self.heightInPixels.getValue(),
        )

    def setImageExtent(self, imageExtent: ImageExtent) -> None:
        self.widthInPixels.setValue(imageExtent.widthInPixels)
        self.heightInPixels.setValue(imageExtent.heightInPixels)

    def getPixelGeometry(self) -> PixelGeometry:
        return PixelGeometry(
            widthInMeters=self.pixelWidthInMeters.getValue(),
            heightInMeters=self.pixelHeightInMeters.getValue(),
        )

    def setPixelGeometry(self, pixelGeometry: PixelGeometry) -> None:
        self.pixelWidthInMeters.setValue(pixelGeometry.widthInMeters)
        self.pixelHeightInMeters.setValue(pixelGeometry.heightInMeters)

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()

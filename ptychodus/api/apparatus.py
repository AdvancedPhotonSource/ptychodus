from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class FresnelZonePlate:
    zonePlateDiameterInMeters: float
    outermostZoneWidthInMeters: float
    centralBeamstopDiameterInMeters: float

    def focalLengthInMeters(self, centralWavelengthInMeters: float) -> float:
        return self.zonePlateDiameterInMeters * self.outermostZoneWidthInMeters \
                / centralWavelengthInMeters


@dataclass(frozen=True)
class ImageExtent:
    widthInPixels: int
    heightInPixels: int

    @property
    def size(self) -> int:
        '''returns the number of pixels in the image'''
        return self.widthInPixels * self.heightInPixels

    @property
    def shape(self) -> tuple[int, int]:
        '''returns the image shape (heightInPixels, widthInPixels) tuple'''
        return self.heightInPixels, self.widthInPixels

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ImageExtent):
            hasSameWidth = (self.widthInPixels == other.widthInPixels)
            hasSameHeight = (self.heightInPixels == other.heightInPixels)
            return (hasSameWidth and hasSameHeight)

        return False

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self.widthInPixels}, {self.heightInPixels})'


@dataclass(frozen=True)
class PixelGeometry:
    widthInMeters: float
    heightInMeters: float

    @classmethod
    def createNull(cls) -> PixelGeometry:
        return cls(0., 0.)

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self.widthInMeters}, {self.heightInMeters})'


@dataclass(frozen=True)
class Detector:
    widthInPixels: int
    heightInPixels: int
    pixelWidthInMeters: float
    pixelHeightInMeters: float
    bitDepth: int

    @property
    def extentInPixels(self) -> ImageExtent:
        return ImageExtent(self.widthInPixels, self.heightInPixels)

    @property
    def pixelGeometry(self) -> PixelGeometry:
        return PixelGeometry(self.pixelWidthInMeters, self.pixelHeightInMeters)

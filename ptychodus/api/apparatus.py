from dataclasses import dataclass
from decimal import Decimal

from .image import ImageExtent


@dataclass(frozen=True)
class FresnelZonePlate:
    zonePlateRadiusInMeters: Decimal
    outermostZoneWidthInMeters: Decimal
    centralBeamstopDiameterInMeters: Decimal

    def focalLengthInMeters(self, centralWavelengthInMeters: Decimal) -> Decimal:
        return 2 * self.zonePlateRadiusInMeters * self.outermostZoneWidthInMeters \
                / centralWavelengthInMeters


@dataclass(frozen=True)
class PixelGeometry:
    widthInMeters: Decimal
    heightInMeters: Decimal


@dataclass(frozen=True)
class Detector:  # FIXME use this
    extentInPixels: ImageExtent
    pixelGeometry: PixelGeometry
    bitDepth: int

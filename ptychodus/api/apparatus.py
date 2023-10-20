from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class FresnelZonePlate:
    zonePlateDiameterInMeters: Decimal
    outermostZoneWidthInMeters: Decimal
    centralBeamstopDiameterInMeters: Decimal

    def focalLengthInMeters(self, centralWavelengthInMeters: Decimal) -> Decimal:
        return self.zonePlateDiameterInMeters * self.outermostZoneWidthInMeters \
                / centralWavelengthInMeters


@dataclass(frozen=True)
class PixelGeometry:
    widthInMeters: Decimal
    heightInMeters: Decimal

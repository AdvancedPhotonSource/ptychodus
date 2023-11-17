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
class PixelGeometry:
    widthInMeters: float
    heightInMeters: float

    @classmethod
    def createNull(cls) -> PixelGeometry:
        return cls(0., 0.)

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class PixelGeometry:
    widthInMeters: Decimal
    heightInMeters: Decimal

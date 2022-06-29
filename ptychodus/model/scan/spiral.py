from __future__ import annotations
from decimal import Decimal
import math

from ..api.scan import ScanInitializer, ScanPoint, ScanPointSequence


class SpiralScanPointSequence(ScanPointSequence):

    def __init__(self, settings: ScanSettings) -> None:
        super().__init__()
        self._settings = settings

    def __getitem__(self, index: int) -> ScanPoint:
        if index >= len(self):
            raise IndexError(f'Index {index} is out of range')

        # theta = omega * t
        # r = a + b * theta
        # x = r * math.cos(theta)
        # y = r * math.sin(theta)

        sqrtIndex = Decimal(index).sqrt()

        # TODO generalize parameters and redo without casting to float
        theta = float(4 * sqrtIndex)
        cosTheta = Decimal(math.cos(theta))
        sinTheta = Decimal(math.sin(theta))

        x = sqrtIndex * cosTheta * self._settings.stepSizeXInMeters.value
        y = sqrtIndex * sinTheta * self._settings.stepSizeYInMeters.value

        return ScanPoint(x, y)

    def __len__(self) -> int:
        nx = self._settings.extentX.value
        ny = self._settings.extentY.value
        return nx * ny


class SpiralScanInitializer(ScanInitializer):

    def __init__(self, settings: ScanSettings) -> None:
        super().__init__()
        self._settings = settings
        self._scanDictionary: dict[str, ScanPointSequence] = {
            'Spiral': SpiralScanPointSequence.createInstance(settings)
        }

    @property
    def simpleName(self) -> str:
        return 'Spiral'

    @property
    def displayName(self) -> str:
        return 'Spiral'

    def __getitem__(self, key: str) -> ScanPointSequence:
        return self._scanDictionary[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._scanDictionary)

    def __len__(self) -> int:
        return len(self._scanDictionary)

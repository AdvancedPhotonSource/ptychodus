from __future__ import annotations

from ..api.scan import ScanInitializer, ScanPoint, ScanPointSequence


class CartesianScanPointSequence(ScanPointSequence):

    def __init__(self, settings: ScanSettings, snake: bool) -> None:
        super().__init__()
        self._settings = settings
        self._snake = snake

    def __getitem__(self, index: int) -> ScanPoint:
        if index >= len(self):
            raise IndexError(f'Index {index} is out of range')

        nx = self._settings.extentX.value
        y, x = divmod(index, nx)

        if self._snake and y & 1:
            x = nx - 1 - x

        xf = x * self._settings.stepSizeXInMeters.value
        yf = y * self._settings.stepSizeYInMeters.value

        return ScanPoint(xf, yf)

    def __len__(self) -> int:
        nx = self._settings.extentX.value
        ny = self._settings.extentY.value
        return nx * ny


class CartesianScanInitializer(ScanInitializer):

    def __init__(self, settings: ScanSettings) -> None:
        super().__init__()
        self._settings = settings
        self._scanDictionary: dict[str, ScanPointSequence] = {
            'Raster': CartesianScanPointSequence(settings, False),
            'Snake': CartesianScanPointSequence(settings, True)
        }

    @property
    def simpleName(self) -> str:
        return 'Cartesian'

    @property
    def displayName(self) -> str:
        return 'Cartesian'

    def __getitem__(self, key: str) -> ScanPointSequence:
        return self._scanDictionary[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._scanDictionary)

    def __len__(self) -> int:
        return len(self._scanDictionary)

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ObjectPixelCenters:
    objectSlice: slice
    patchCoordinates: Sequence[float]


@dataclass(frozen=True)
class ObjectPatchAxis:
    parent: ObjectAxis
    centerInMeters: float
    numberOfPixels: int

    @property
    def _radius(self) -> float:
        return self.numberOfPixels / 2

    @property
    def _shift(self) -> float:
        dx = (self.centerInMeters - self.parent.centerInMeters) / self.parent.pixelSizeInMeters
        dn = (self.numberOfPixels - self.parent.numberOfPixels) / 2
        return dx - dn

    def getObjectCoordinates(self) -> Sequence[float]:
        shift = self._shift
        return [idx + shift for idx in range(self.numberOfPixels)]

    def getObjectPixelCenters(self) -> ObjectPixelCenters:
        '''returns object pixel centers covered by this patch'''
        patchCenter = self.parent.mapScanCoordinateToObjectCoordinate(self.centerInMeters)
        first = int(patchCenter - self._radius + 0.5)
        last = int(patchCenter + self._radius + 0.5)
        delta = 0.5 - self._shift  # add 1/2 for pixel center, shift to patch coordinates

        return ObjectPixelCenters(
            objectSlice=slice(first, last),
            patchCoordinates=[pixelIndex + delta for pixelIndex in range(first, last)],
        )

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import numpy
import numpy.typing

from .geometry import Point2D
from .image import ImageExtent
from .scan import ScanPoint

ObjectArrayType: TypeAlias = numpy.typing.NDArray[numpy.complexfloating[Any, Any]]

# object point coordinates are conventionally in pixel units
ObjectPoint: TypeAlias = Point2D[float]


@dataclass(frozen=True)
class ObjectAxis:
    centerInMeters: float
    pixelSizeInMeters: float
    numberOfPixels: int

    @property
    def _radius(self) -> float:
        return self.numberOfPixels / 2

    def mapObjectCoordinateToScanCoordinate(self, objectCoordinate: float) -> float:
        '''maps an object coordinate to a scan coordinate'''
        return self.centerInMeters + self.pixelSizeInMeters * (objectCoordinate - self._radius)

    def mapScanCoordinateToObjectCoordinate(self, scanCoordinate: float) -> float:
        '''maps a scan coordinate to an object coordinate'''
        return (scanCoordinate - self.centerInMeters) / self.pixelSizeInMeters + self._radius


@dataclass(frozen=True)
class ObjectGrid:
    axisX: ObjectAxis
    axisY: ObjectAxis

    def mapObjectPointToScanPoint(self, point: ObjectPoint) -> ScanPoint:
        x = self.axisX.mapObjectCoordinateToScanCoordinate(point.x)
        y = self.axisY.mapObjectCoordinateToScanCoordinate(point.y)
        return ScanPoint(x, y)

    def mapScanPointToObjectPoint(self, point: ScanPoint) -> ObjectPoint:
        x = self.axisX.mapScanCoordinateToObjectCoordinate(point.x)
        y = self.axisY.mapScanCoordinateToObjectCoordinate(point.y)
        return ObjectPoint(x, y)


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

    def getPixelScanCoordinates(self) -> Sequence[float]:
        u = self.centerInMeters - self._radius
        return [idx + u for idx in range(self.numberOfPixels)]

    @property
    def _shift(self) -> float:
        dx = (self.centerInMeters - self.parent.centerInMeters) / self.parent.pixelSizeInMeters
        dn = (self.numberOfPixels - self.parent.numberOfPixels) / 2
        return dx - dn

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


@dataclass(frozen=True)
class ObjectPatch:
    axisX: ObjectPatchAxis
    axisY: ObjectPatchAxis
    array: ObjectArrayType


class ObjectInterpolator(ABC):

    @abstractmethod
    def getGrid(self) -> ObjectGrid:
        '''returns the object coordinate grid'''
        pass

    @abstractmethod
    def getArray(self) -> ObjectArrayType:
        '''returns the object array'''
        pass

    @abstractmethod
    def getPatch(self, patchCenter: ScanPoint, patchExtent: ImageExtent) -> ObjectPatch:
        '''returns an interpolated patch from the object array'''
        pass


class ObjectPhaseCenteringStrategy(ABC):
    '''interface for object phase centering strategies'''

    @abstractmethod
    def __call__(self, array: ObjectArrayType) -> ObjectArrayType:
        '''returns the phase-centered array'''
        pass


class ObjectFileReader(ABC):

    @abstractmethod
    def read(self, filePath: Path) -> ObjectArrayType:
        '''reads an object from file'''
        pass


class ObjectFileWriter(ABC):

    @abstractmethod
    def write(self, filePath: Path, array: ObjectArrayType) -> None:
        '''writes an object to file'''
        pass

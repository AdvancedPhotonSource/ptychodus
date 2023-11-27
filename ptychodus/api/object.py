from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import numpy
import numpy.typing

from .apparatus import ImageExtent, PixelGeometry
from .geometry import Point2D
from .scan import ScanPoint

# object point coordinates are conventionally in pixel units
ObjectPoint: TypeAlias = Point2D[float]
ObjectArrayType: TypeAlias = numpy.typing.NDArray[numpy.complexfloating[Any, Any]]


class Object:

    def __init__(self, array: ObjectArrayType | None = None) -> None:
        self._array = numpy.zeros((1, 0, 0), dtype=complex)
        self._layerDistanceInMeters = [numpy.inf]
        self._centerXInMeters = 0.
        self._centerYInMeters = 0.

        if array is not None:
            self.setArray(array)

    def copy(self) -> Object:
        clone = Object()
        clone._array = self._array.copy()
        clone._layerDistanceInMeters = self._layerDistanceInMeters.copy()
        clone._centerXInMeters = float(self._centerXInMeters)
        clone._centerYInMeters = float(self._centerYInMeters)
        return clone

    def getArray(self) -> ObjectArrayType:
        return self._array

    def setArray(self, array: ObjectArrayType) -> None:
        if not numpy.iscomplexobj(array):
            raise TypeError('Object must be a complex-valued ndarray')

        if array.ndim == 2:
            self._array = array[numpy.newaxis, :, :]
        elif array.ndim == 3:
            self._array = array
        else:
            raise ValueError('Object must be 2- or 3-dimensional ndarray.')

        numberOfAddedLayers = self._array.shape[-3] - len(self._layerDistanceInMeters)

        if numberOfAddedLayers > 0:
            self._layerDistanceInMeters[-1] = 0.
            self._layerDistanceInMeters.extend([0.] * numberOfAddedLayers)
            self._layerDistanceInMeters[-1] = numpy.inf

    def getDataType(self) -> numpy.dtype:
        return self._array.dtype

    def getImageExtent(self) -> ImageExtent:
        return ImageExtent(
            widthInPixels=self._array.shape[-1],
            heightInPixels=self._array.shape[-2],
        )

    def getSizeInBytes(self) -> int:
        return self._array.nbytes

    def getNumberOfLayers(self) -> int:
        return self._array.shape[-3]

    def getLayer(self, number: int) -> ObjectArrayType:
        return self._array[number, :, :]

    def getLayersFlattened(self) -> ObjectArrayType:
        return numpy.prod(self._array, axis=-3)

    def getLayerDistancesInMeters(self) -> Sequence[float]:
        return self._layerDistanceInMeters

    def getLayerDistanceInMeters(self, number: int) -> float:
        return self._layerDistanceInMeters[number]

    def setLayerDistanceInMeters(self, layer: int, distance: float) -> None:
        if 0 <= layer and layer < self.getNumberOfLayers() - 1:
            self._layerDistanceInMeters[layer] = distance

    def getCenter(self) -> ScanPoint:
        return ScanPoint(self._centerXInMeters, self._centerYInMeters)

    def setCenter(self, center: ScanPoint) -> None:
        self._centerXInMeters = center.x
        self._centerYInMeters = center.y

    def hasSameShape(self, other: Object) -> bool:
        return (self._array.shape == other._array.shape
                and self._layerDistanceInMeters == other._layerDistanceInMeters)


@dataclass(frozen=True)
class ObjectAxis:
    centerInMeters: float
    numberOfPixels: int
    pixelSizeInMeters: float

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

    @classmethod
    def createInstance(cls, midpoint: ScanPoint, extent: ImageExtent,
                       pixelGeometry: PixelGeometry) -> ObjectGrid:
        return cls(
            axisX=ObjectAxis(
                centerInMeters=midpoint.x,
                numberOfPixels=extent.widthInPixels,
                pixelSizeInMeters=float(pixelGeometry.widthInMeters),
            ),
            axisY=ObjectAxis(
                centerInMeters=midpoint.y,
                numberOfPixels=extent.heightInPixels,
                pixelSizeInMeters=float(pixelGeometry.heightInMeters),
            ),
        )

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


@dataclass(frozen=True)
class ObjectPatchGrid:
    axisX: ObjectPatchAxis
    axisY: ObjectPatchAxis

    @classmethod
    def createInstance(cls, parent: ObjectGrid, patchCenter: ScanPoint,
                       patchExtent: ImageExtent) -> ObjectPatchGrid:
        return cls(
            axisX=ObjectPatchAxis(parent.axisX, patchCenter.x, patchExtent.widthInPixels),
            axisY=ObjectPatchAxis(parent.axisY, patchCenter.y, patchExtent.heightInPixels),
        )


@dataclass(frozen=True)
class ObjectPatch:
    grid: ObjectPatchGrid
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
    def read(self, filePath: Path) -> Object:
        '''reads an object from file'''
        pass


class ObjectFileWriter(ABC):

    @abstractmethod
    def write(self, filePath: Path, object_: Object) -> None:
        '''writes an object to file'''
        pass

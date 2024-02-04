from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import numpy
import numpy.typing

from .geometry import Point2D
from .patterns import ImageExtent, PixelGeometry

ObjectArrayType: TypeAlias = numpy.typing.NDArray[numpy.complexfloating[Any, Any]]


@dataclass(frozen=True)
class ObjectGeometry:
    widthInPixels: int
    heightInPixels: int
    pixelWidthInMeters: float
    pixelHeightInMeters: float
    centerXInMeters: float
    centerYInMeters: float

    @property
    def widthInMeters(self) -> float:
        return self.widthInPixels * self.pixelWidthInMeters

    @property
    def heightInMeters(self) -> float:
        return self.heightInPixels * self.pixelHeightInMeters

    @property
    def minimumXInMeters(self) -> float:
        return self.centerXInMeters - self.widthInMeters / 2.

    @property
    def minimumYInMeters(self) -> float:
        return self.centerYInMeters - self.heightInMeters / 2.

    @property
    def _radiusX(self) -> float:
        return self.widthInPixels / 2

    @property
    def _radiusY(self) -> float:
        return self.heightInPixels / 2

    def mapObjectPointToScanPoint(self, objectPoint: Point2D) -> Point2D:
        x = self.centerXInMeters + self.pixelWidthInMeters * (objectPoint.x - self._radiusX)
        y = self.centerYInMeters + self.pixelHeightInMeters * (objectPoint.y - self._radiusY)
        return Point2D(x, y)

    def mapScanPointToObjectPoint(self, scanPoint: Point2D) -> Point2D:
        x = (scanPoint.x - self.centerXInMeters) / self.pixelWidthInMeters + self._radiusX
        y = (scanPoint.y - self.centerYInMeters) / self.pixelHeightInMeters + self._radiusY
        return Point2D(x, y)

    def contains(self, geometry: ObjectGeometry) -> bool:
        dx = self.centerXInMeters - geometry.centerXInMeters
        dy = self.centerYInMeters - geometry.centerYInMeters
        dw = self.widthInMeters - geometry.widthInMeters
        dh = self.heightInMeters - geometry.heightInMeters
        return (abs(dx) <= dw and abs(dy) <= dh)


class ObjectGeometryProvider(ABC):

    @abstractmethod
    def getObjectGeometry(self) -> ObjectGeometry:
        pass


class Object:

    def __init__(self,
                 array: ObjectArrayType | None = None,
                 layerDistanceInMeters: Sequence[float] | None = None,
                 *,
                 pixelWidthInMeters: float = 0.,
                 pixelHeightInMeters: float = 0.,
                 centerXInMeters: float = 0.,
                 centerYInMeters: float = 0.) -> None:
        if array is None:
            self._array = numpy.zeros((1, 0, 0), dtype=complex)
        else:
            if numpy.iscomplexobj(array):
                if array.ndim == 2:
                    self._array = array[numpy.newaxis, :, :]
                elif array.ndim == 3:
                    self._array = array
                else:
                    raise ValueError('Object must be 2- or 3-dimensional ndarray.')
            else:
                raise TypeError('Object must be a complex-valued ndarray')

        if layerDistanceInMeters is None:
            self._layerDistanceInMeters: Sequence[float] = [numpy.inf]
        else:
            self._layerDistanceInMeters = layerDistanceInMeters

        expectedLayers = self.numberOfLayers
        actualLayers = len(self._layerDistanceInMeters)

        if actualLayers != expectedLayers:
            raise ValueError(f'Expected {expectedLayers} layer distances; got {actualLayers}!')

        self._pixelWidthInMeters = pixelWidthInMeters
        self._pixelHeightInMeters = pixelHeightInMeters
        self._centerXInMeters = centerXInMeters
        self._centerYInMeters = centerYInMeters

    @property
    def array(self) -> ObjectArrayType:
        return self._array

    @property
    def dataType(self) -> numpy.dtype:
        return self._array.dtype

    @property
    def numberOfLayers(self) -> int:
        return self._array.shape[-3]

    @property
    def sizeInBytes(self) -> int:
        return self._array.nbytes

    def getGeometry(self) -> ObjectGeometry:
        return ObjectGeometry(
            widthInPixels=self._array.shape[-1],
            heightInPixels=self._array.shape[-2],
            pixelWidthInMeters=self._pixelWidthInMeters,
            pixelHeightInMeters=self._pixelHeightInMeters,
            centerXInMeters=self._centerXInMeters,
            centerYInMeters=self._centerYInMeters,
        )

    def getPixelGeometry(self) -> PixelGeometry:
        return PixelGeometry(
            widthInMeters=self._pixelWidthInMeters,
            heightInMeters=self._pixelHeightInMeters,
        )

    def getLayer(self, number: int) -> ObjectArrayType:
        return self._array[number, :, :]

    def getLayersFlattened(self) -> ObjectArrayType:
        return numpy.prod(self._array, axis=-3)

    @property
    def layerDistanceInMeters(self) -> Sequence[float]:
        return self._layerDistanceInMeters

    def getLayerDistanceInMeters(self, number: int) -> float:
        return self._layerDistanceInMeters[number]


class ObjectInterpolator(ABC):

    @abstractmethod
    def getPatch(self, patchCenter: Point2D, patchExtent: ImageExtent) -> Object:
        '''returns an interpolated patch from the object array'''
        pass


class ObjectPhaseCenteringStrategy(ABC):

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

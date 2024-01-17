from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypeAlias

import numpy
import numpy.typing

from .geometry import Point2D
from .patterns import ImageExtent

ObjectArrayType: TypeAlias = numpy.typing.NDArray[numpy.complexfloating[Any, Any]]


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
    def heightInPixels(self) -> int:
        return self._array.shape[-2]

    @property
    def widthInPixels(self) -> int:
        return self._array.shape[-1]

    @property
    def sizeInBytes(self) -> int:
        return self._array.nbytes

    @property
    def pixelWidthInMeters(self) -> float:
        return self._pixelWidthInMeters

    @property
    def pixelHeightInMeters(self) -> float:
        return self._pixelHeightInMeters

    def getLayer(self, number: int) -> ObjectArrayType:
        return self._array[number, :, :]

    def getLayersFlattened(self) -> ObjectArrayType:
        return numpy.prod(self._array, axis=-3)

    @property
    def layerDistanceInMeters(self) -> Sequence[float]:
        return self._layerDistanceInMeters

    def getLayerDistanceInMeters(self, number: int) -> float:
        return self._layerDistanceInMeters[number]

    @property
    def centerXInMeters(self) -> float:
        return self._centerXInMeters

    @property
    def centerYInMeters(self) -> float:
        return self._centerYInMeters

    def getCenter(self) -> Point2D:
        return Point2D(self._centerXInMeters, self._centerYInMeters)

    @property
    def _radiusX(self) -> float:
        return self.widthInPixels / 2

    @property
    def _radiusY(self) -> float:
        return self.heightInPixels / 2

    def mapObjectPointToScanPoint(self, objectPoint: Point2D) -> Point2D:
        x = self._centerXInMeters + self._pixelWidthInMeters * (objectPoint.x - self._radiusX)
        y = self._centerYInMeters + self._pixelHeightInMeters * (objectPoint.y - self._radiusY)
        return Point2D(x, y)

    def mapScanPointToObjectPoint(self, scanPoint: Point2D) -> Point2D:
        x = (scanPoint.x - self._centerXInMeters) / self._pixelWidthInMeters + self._radiusX
        y = (scanPoint.y - self._centerYInMeters) / self._pixelHeightInMeters + self._radiusY
        return Point2D(x, y)


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

from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import numpy
import numpy.typing

from .geometry import ImageExtent, PixelGeometry
from .scan import ScanPoint

ObjectArrayType: TypeAlias = numpy.typing.NDArray[numpy.complexfloating[Any, Any]]


@dataclass(frozen=True)
class ObjectPoint:
    index: int
    positionXInPixels: float
    positionYInPixels: float


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
        return self.centerXInMeters - self.widthInMeters / 2.0

    @property
    def minimumYInMeters(self) -> float:
        return self.centerYInMeters - self.heightInMeters / 2.0

    def getPixelGeometry(self) -> PixelGeometry:
        return PixelGeometry(
            widthInMeters=self.pixelWidthInMeters,
            heightInMeters=self.pixelHeightInMeters,
        )

    def mapObjectPointToScanPoint(self, point: ObjectPoint) -> ScanPoint:
        rx_px = self.widthInPixels / 2
        ry_px = self.heightInPixels / 2
        dx_m = self.pixelWidthInMeters
        dy_m = self.pixelHeightInMeters

        x_m = self.centerXInMeters + dx_m * (point.positionXInPixels - rx_px)
        y_m = self.centerYInMeters + dy_m * (point.positionYInPixels - ry_px)

        return ScanPoint(point.index, x_m, y_m)

    def mapScanPointToObjectPoint(self, point: ScanPoint) -> ObjectPoint:
        rx_px = self.widthInPixels / 2
        ry_px = self.heightInPixels / 2
        dx_m = self.pixelWidthInMeters
        dy_m = self.pixelHeightInMeters

        x_px = (point.positionXInMeters - self.centerXInMeters) / dx_m + rx_px
        y_px = (point.positionYInMeters - self.centerYInMeters) / dy_m + ry_px

        return ObjectPoint(point.index, x_px, y_px)

    def contains(self, geometry: ObjectGeometry) -> bool:
        dx = self.centerXInMeters - geometry.centerXInMeters
        dy = self.centerYInMeters - geometry.centerYInMeters
        dw = self.widthInMeters - geometry.widthInMeters
        dh = self.heightInMeters - geometry.heightInMeters
        return abs(dx) <= dw and abs(dy) <= dh


class ObjectGeometryProvider(ABC):
    @abstractmethod
    def getObjectGeometry(self) -> ObjectGeometry:
        pass


class Object:
    def __init__(
        self,
        array: ObjectArrayType | None,
        pixelGeometry: PixelGeometry | None,
        layerDistanceInMeters: Sequence[float] = [],
        *,
        centerXInMeters: float = 0.0,
        centerYInMeters: float = 0.0,
    ) -> None:
        if array is None:
            self._array = numpy.zeros((1, 0, 0), dtype=complex)
        elif numpy.iscomplexobj(array):
            match array.ndim:
                case 2:
                    self._array = array[numpy.newaxis, ...]
                case 3:
                    self._array = array
                case _:
                    raise ValueError('Object must be 2- or 3-dimensional ndarray.')
        else:
            raise TypeError('Object must be a complex-valued ndarray')

        self._pixelGeometry = pixelGeometry
        self._layerDistanceInMeters = layerDistanceInMeters
        self._centerXInMeters = centerXInMeters
        self._centerYInMeters = centerYInMeters

        expectedLayers = self._array.shape[-3]
        actualLayers = len(layerDistanceInMeters) + 1

        if actualLayers != expectedLayers:
            raise ValueError(f'Expected {expectedLayers} layer distances; got {actualLayers}!')

    def copy(self) -> Object:
        return Object(
            array=self._array.copy(),
            pixelGeometry=None if self._pixelGeometry is None else self._pixelGeometry.copy(),
            layerDistanceInMeters=list(self._layerDistanceInMeters),
            centerXInMeters=float(self._centerXInMeters),
            centerYInMeters=float(self._centerYInMeters),
        )

    def getArray(self) -> ObjectArrayType:
        return self._array

    @property
    def dataType(self) -> numpy.dtype:
        return self._array.dtype

    @property
    def sizeInBytes(self) -> int:
        return self._array.nbytes

    @property
    def widthInPixels(self) -> int:
        return self._array.shape[-1]

    @property
    def heightInPixels(self) -> int:
        return self._array.shape[-2]

    @property
    def numberOfLayers(self) -> int:
        return self._array.shape[-3]

    def getPixelGeometry(self) -> PixelGeometry | None:
        return self._pixelGeometry

    @property
    def centerXInMeters(self) -> float:
        return self._centerXInMeters

    @property
    def centerYInMeters(self) -> float:
        return self._centerYInMeters

    def getGeometry(self) -> ObjectGeometry:
        pixelWidthInMeters = 0.0
        pixelHeightInMeters = 0.0

        if self._pixelGeometry is not None:
            pixelWidthInMeters = self._pixelGeometry.widthInMeters
            pixelHeightInMeters = self._pixelGeometry.heightInMeters

        return ObjectGeometry(
            widthInPixels=self.widthInPixels,
            heightInPixels=self.heightInPixels,
            pixelWidthInMeters=pixelWidthInMeters,
            pixelHeightInMeters=pixelHeightInMeters,
            centerXInMeters=self._centerXInMeters,
            centerYInMeters=self._centerYInMeters,
        )

    def getLayer(self, number: int) -> ObjectArrayType:
        return self._array[number, :, :]

    def getLayersFlattened(self) -> ObjectArrayType:
        return numpy.prod(self._array, axis=-3)

    @property
    def layerDistanceInMeters(self) -> Sequence[float]:
        return self._layerDistanceInMeters

    def getTotalLayerDistanceInMeters(self) -> float:
        return sum(self._layerDistanceInMeters)


class ObjectInterpolator(ABC):
    @abstractmethod
    def getPatch(self, patchCenter: ScanPoint, patchExtent: ImageExtent) -> Object:
        """returns an interpolated patch from the object array"""
        pass


class ObjectPhaseCenteringStrategy(ABC):
    @abstractmethod
    def __call__(self, array: ObjectArrayType) -> ObjectArrayType:
        """returns the phase-centered array"""
        pass


class ObjectFileReader(ABC):
    @abstractmethod
    def read(self, filePath: Path) -> Object:
        """reads an object from file"""
        pass


class ObjectFileWriter(ABC):
    @abstractmethod
    def write(self, filePath: Path, object_: Object) -> None:
        """writes an object to file"""
        pass

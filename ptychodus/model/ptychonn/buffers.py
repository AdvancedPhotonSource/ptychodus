from __future__ import annotations
import logging

import numpy
import numpy.typing

from ptychodus.api.geometry import ImageExtent
from ptychodus.api.object import ObjectArrayType
from ptychodus.api.typing import Float32ArrayType

logger = logging.getLogger(__name__)


class PatternCircularBuffer:

    def __init__(self, extent: ImageExtent, maxSize: int) -> None:
        self._buffer: Float32ArrayType = numpy.zeros(
            (maxSize, *extent.shape),
            dtype=numpy.float32,
        )
        self._pos = 0
        self._full = False

    @classmethod
    def createZeroSized(cls) -> PatternCircularBuffer:
        return cls(ImageExtent(0, 0), 0)

    @property
    def isZeroSized(self) -> bool:
        return self._buffer.size == 0

    def append(self, array: Float32ArrayType) -> None:
        self._buffer[self._pos, :, :] = array
        self._pos += 1

        if self._pos == self._buffer.shape[0]:
            self._pos = 0
            self._full = True

    def getBuffer(self) -> Float32ArrayType:
        return self._buffer if self._full else self._buffer[:self._pos]

    def setBuffer(self, array: Float32ArrayType) -> None:
        self._buffer = array
        self._pos = 0
        self._full = True


class ObjectPatchCircularBuffer:

    def __init__(self, extent: ImageExtent, channels: int, maxSize: int) -> None:
        self._buffer: Float32ArrayType = numpy.zeros(
            (maxSize, channels, *extent.shape),
            dtype=numpy.float32,
        )
        self._pos = 0
        self._full = False

    @classmethod
    def createZeroSized(cls) -> ObjectPatchCircularBuffer:
        return cls(ImageExtent(0, 0), 0, 0)

    @property
    def isZeroSized(self) -> bool:
        return self._buffer.size == 0

    def append(self, array: ObjectArrayType) -> None:
        self._buffer[self._pos, 0, :, :] = numpy.angle(array).astype(numpy.float32)

        if self._buffer.shape[1] > 1:
            self._buffer[self._pos, 1, :, :] = numpy.absolute(array).astype(numpy.float32)

        self._pos += 1

        if self._pos == self._buffer.shape[0]:
            self._pos = 0
            self._full = True

    def getBuffer(self) -> Float32ArrayType:
        return self._buffer if self._full else self._buffer[:self._pos]

    def setBuffer(self, array: Float32ArrayType) -> None:
        self._buffer = array
        self._pos = 0
        self._full = True

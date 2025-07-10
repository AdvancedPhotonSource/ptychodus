from __future__ import annotations
from typing import TypeAlias
import logging

import numpy
import numpy.typing

from ptychodus.api.geometry import ImageExtent
from ptychodus.api.typing import ComplexArrayType

Float32ArrayType: TypeAlias = numpy.typing.NDArray[numpy.float32]

logger = logging.getLogger(__name__)


class PatternCircularBuffer:
    def __init__(self, extent: ImageExtent, max_size: int) -> None:
        self._buffer: Float32ArrayType = numpy.zeros(
            (max_size, *extent.shape),
            dtype=numpy.float32,
        )
        self._pos = 0
        self._full = False

    @classmethod
    def create_zero_sized(cls) -> PatternCircularBuffer:
        return cls(ImageExtent(0, 0), 0)

    @property
    def is_zero_sized(self) -> bool:
        return self._buffer.size == 0

    def append(self, array: Float32ArrayType) -> None:
        self._buffer[self._pos, :, :] = array
        self._pos += 1

        if self._pos == self._buffer.shape[0]:
            self._pos = 0
            self._full = True

    def get_buffer(self) -> Float32ArrayType:
        return self._buffer if self._full else self._buffer[: self._pos]

    def set_buffer(self, array: Float32ArrayType) -> None:
        self._buffer = array
        self._pos = 0
        self._full = True


class ObjectPatchCircularBuffer:
    def __init__(self, extent: ImageExtent, channels: int, max_size: int) -> None:
        self._buffer: Float32ArrayType = numpy.zeros(
            (max_size, channels, *extent.shape),
            dtype=numpy.float32,
        )
        self._pos = 0
        self._full = False

    @classmethod
    def create_zero_sized(cls) -> ObjectPatchCircularBuffer:
        return cls(ImageExtent(0, 0), 0, 0)

    @property
    def is_zero_sized(self) -> bool:
        return self._buffer.size == 0

    def append(self, array: ComplexArrayType) -> None:
        self._buffer[self._pos, 0, :, :] = numpy.angle(array).astype(numpy.float32)

        if self._buffer.shape[1] > 1:
            self._buffer[self._pos, 1, :, :] = numpy.absolute(array).astype(numpy.float32)

        self._pos += 1

        if self._pos == self._buffer.shape[0]:
            self._pos = 0
            self._full = True

    def get_buffer(self) -> Float32ArrayType:
        return self._buffer if self._full else self._buffer[: self._pos]

    def set_buffer(self, array: Float32ArrayType) -> None:
        self._buffer = array
        self._pos = 0
        self._full = True

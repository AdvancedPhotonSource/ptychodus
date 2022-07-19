from __future__ import annotations
from abc import ABC, abstractmethod, abstractproperty
from collections.abc import Callable
from dataclasses import dataclass

import numpy
import numpy.typing

RealArrayType = numpy.typing.NDArray[numpy.floating]


@dataclass(frozen=True)
class ImageExtent:
    width: int
    height: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    def __add__(self, other: ImageExtent) -> ImageExtent:
        if isinstance(other, ImageExtent):
            w = self.width + other.width
            h = self.height + other.height
            return ImageExtent(width=w, height=h)

    def __sub__(self, other: ImageExtent) -> ImageExtent:
        if isinstance(other, ImageExtent):
            w = self.width - other.width
            h = self.height - other.height
            return ImageExtent(width=w, height=h)

    def __mul__(self, other: int) -> ImageExtent:
        if isinstance(other, int):
            w = self.width * other
            h = self.height * other
            return ImageExtent(width=w, height=h)

    def __rmul__(self, other: int) -> ImageExtent:
        if isinstance(other, int):
            w = other * self.width
            h = other * self.height
            return ImageExtent(width=w, height=h)

    def __floordiv__(self, other: int) -> ImageExtent:
        if isinstance(other, int):
            w = self.width // other
            h = self.height // other
            return ImageExtent(width=w, height=h)

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self.width}, {self.height})'


class ScalarTransformation(Callable[[RealArrayType], RealArrayType]):

    @abstractproperty
    def name(self) -> str:
        pass

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy
import numpy.typing

RealArrayType: TypeAlias = numpy.typing.NDArray[numpy.floating[Any]]


@dataclass(frozen=True)
class ImageExtent:
    '''image extent in pixels'''
    width: int
    height: int

    @property
    def size(self) -> int:
        '''returns the number of pixels in the image'''
        return self.width * self.height

    @property
    def shape(self) -> tuple[int, int]:
        '''returns the image shape (height, width) tuple'''
        return self.height, self.width

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ImageExtent):
            return (self.width == other.width and self.height == other.height)

        return False

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


class ScalarTransformation(ABC):
    '''interface for real-valued transformations of a real array'''

    @abstractmethod
    def decorateText(self, text: str) -> str:
        pass

    @abstractmethod
    def __call__(self, array: RealArrayType) -> RealArrayType:
        '''returns the transformed input array'''
        pass

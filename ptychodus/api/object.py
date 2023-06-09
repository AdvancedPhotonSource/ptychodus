from abc import ABC, abstractmethod
from decimal import Decimal
from pathlib import Path
from typing import Any, TypeAlias

import numpy
import numpy.typing

from .geometry import Point2D

ObjectArrayType: TypeAlias = numpy.typing.NDArray[numpy.complexfloating[Any, Any]]

# object point coordinates are conventionally in pixel units
ObjectPoint: TypeAlias = Point2D[Decimal]


class ObjectPhaseCenteringStrategy(ABC):
    '''interface for object phase centering strategies'''

    @property
    @abstractmethod
    def name(self) -> str:
        '''returns a unique name'''
        pass

    @abstractmethod
    def __call__(self, array: ObjectArrayType) -> ObjectArrayType:
        '''returns the phase-centered array'''
        pass


class ObjectFileReader(ABC):

    @property
    @abstractmethod
    def simpleName(self) -> str:
        '''returns a unique name that is appropriate for a settings file'''
        pass

    @property
    @abstractmethod
    def fileFilter(self) -> str:
        '''returns a unique name that is prettified for visual display'''
        pass

    @abstractmethod
    def read(self, filePath: Path) -> ObjectArrayType:
        '''reads an object from file'''
        pass


class ObjectFileWriter(ABC):

    @property
    @abstractmethod
    def simpleName(self) -> str:
        '''returns a unique name that is appropriate for a settings file'''
        pass

    @property
    @abstractmethod
    def fileFilter(self) -> str:
        '''returns a unique name that is prettified for visual display'''
        pass

    @abstractmethod
    def write(self, filePath: Path, array: ObjectArrayType) -> None:
        '''writes an object to file'''
        pass

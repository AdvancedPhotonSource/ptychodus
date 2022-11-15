from abc import ABC, abstractmethod, abstractproperty
from pathlib import Path

import numpy
import numpy.typing

ObjectArrayType = numpy.typing.NDArray[numpy.complexfloating]


class ObjectFileReader(ABC):

    @abstractproperty
    def simpleName(self) -> str:
        '''returns a unique name that is appropriate for a settings file'''
        pass

    @abstractproperty
    def fileFilter(self) -> str:
        '''returns a unique name that is prettified for visual display'''
        pass

    @abstractmethod
    def read(self, filePath: Path) -> ObjectArrayType:
        '''reads an object from file'''
        pass


class ObjectFileWriter(ABC):

    @abstractproperty
    def simpleName(self) -> str:
        '''returns a unique name that is appropriate for a settings file'''
        pass

    @abstractproperty
    def fileFilter(self) -> str:
        '''returns a unique name that is prettified for visual display'''
        pass

    @abstractmethod
    def write(self, filePath: Path, array: ObjectArrayType) -> None:
        '''writes an object to file'''
        pass

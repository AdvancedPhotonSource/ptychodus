from abc import ABC, abstractmethod, abstractproperty
from pathlib import Path
from typing import Any

import numpy
import numpy.typing

ProbeArrayType = numpy.typing.NDArray[numpy.complexfloating[Any, Any]]


class ProbeFileReader(ABC):

    @abstractproperty
    def simpleName(self) -> str:
        '''returns a unique name that is appropriate for a settings file'''
        pass

    @abstractproperty
    def fileFilter(self) -> str:
        '''returns a unique name that is prettified for visual display'''
        pass

    @abstractmethod
    def read(self, filePath: Path) -> ProbeArrayType:
        '''reads a probe from file'''
        pass


class ProbeFileWriter(ABC):

    @abstractproperty
    def simpleName(self) -> str:
        '''returns a unique name that is appropriate for a settings file'''
        pass

    @abstractproperty
    def fileFilter(self) -> str:
        '''returns a unique name that is prettified for visual display'''
        pass

    @abstractmethod
    def write(self, filePath: Path, array: ProbeArrayType) -> None:
        '''writes a probe to file'''
        pass

from abc import ABC, abstractmethod, abstractproperty
from pathlib import Path
from typing import Callable

import numpy
import numpy.typing

ObjectArrayType = numpy.typing.NDArray[numpy.complexfloating]
ObjectInitializerType = Callable[[], ObjectArrayType]


class ObjectFileReader(ABC):
    @abstractproperty
    def simpleName(self) -> str:
        pass

    @abstractproperty
    def fileFilter(self) -> str:
        pass

    @abstractmethod
    def read(self, filePath: Path) -> ObjectArrayType:
        pass


class ObjectFileWriter(ABC):
    @abstractproperty
    def simpleName(self) -> str:
        pass

    @abstractproperty
    def fileFilter(self) -> str:
        pass

    @abstractmethod
    def write(self, filePath: Path, array: ObjectArrayType) -> None:
        pass

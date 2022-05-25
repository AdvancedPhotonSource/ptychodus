from abc import ABC, abstractmethod, abstractproperty
from pathlib import Path
from typing import Callable

import numpy
import numpy.typing

ProbeArrayType = numpy.typing.NDArray[numpy.complexfloating]
ProbeInitializerType = Callable[[], ProbeArrayType]


class ProbeFileReader(ABC):

    @abstractproperty
    def simpleName(self) -> str:
        pass

    @abstractproperty
    def fileFilter(self) -> str:
        pass

    @abstractmethod
    def read(self, filePath: Path) -> ProbeArrayType:
        pass


class ProbeFileWriter(ABC):

    @abstractproperty
    def simpleName(self) -> str:
        pass

    @abstractproperty
    def fileFilter(self) -> str:
        pass

    @abstractmethod
    def write(self, filePath: Path, array: ProbeArrayType) -> None:
        pass

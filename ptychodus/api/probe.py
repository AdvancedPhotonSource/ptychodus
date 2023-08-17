from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TypeAlias

import numpy
import numpy.typing

ProbeArrayType: TypeAlias = numpy.typing.NDArray[numpy.complexfloating[Any, Any]]


class ProbeFileReader(ABC):

    @abstractmethod
    def read(self, filePath: Path) -> ProbeArrayType:
        '''reads a probe from file'''
        pass


class ProbeFileWriter(ABC):

    @abstractmethod
    def write(self, filePath: Path, array: ProbeArrayType) -> None:
        '''writes a probe to file'''
        pass

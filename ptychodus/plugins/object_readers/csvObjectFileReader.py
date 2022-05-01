from pathlib import Path

import numpy

from ptychodus.api.object import ObjectFileReader, ObjectArrayType


class CSVObjectFileReader(ObjectFileReader):
    @property
    def simpleName(self) -> str:
        return 'CSV'

    @property
    def fileFilter(self) -> str:
        return 'Comma-Separated Values Files (*.csv)'

    def read(self, filePath: Path) -> ObjectArrayType:
        return numpy.genfromtxt(filePath, delimiter=',', dtype='complex')


def registrable_plugins() -> list[ObjectFileReader]:
    return [CSVObjectFileReader()]

from pathlib import Path

import numpy

from ptychodus.api.probe import ProbeFileReader, ProbeArrayType


class NPYProbeFileReader(ProbeFileReader):
    @property
    def simpleName(self) -> str:
        return 'NPY'

    @property
    def fileFilter(self) -> str:
        return 'NumPy Binary Files (*.npy)'

    def read(self, filePath: Path) -> ProbeArrayType:
        return numpy.load(filePath)


def registrable_plugins() -> list[ProbeFileReader]:
    return [NPYProbeFileReader()]

from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias
import logging

import numpy
import numpy.typing

from ...api.parametric import ParameterRepository
from ...api.probe import Probe, ProbeArrayType, ProbeFileReader, ProbeGeometry

logger = logging.getLogger(__name__)

RealArrayType: TypeAlias = numpy.typing.NDArray[numpy.floating[Any]]


@dataclass(frozen=True)
class ProbeTransverseCoordinates:
    positionXInMeters: RealArrayType
    positionYInMeters: RealArrayType

    @property
    def positionRInMeters(self) -> RealArrayType:
        return numpy.hypot(self.positionXInMeters, self.positionYInMeters)


class ProbeBuilder(ParameterRepository):

    def __init__(self, name: str) -> None:
        super().__init__('Builder')
        self._name = self._registerStringParameter('Name', name)

    def getTransverseCoordinates(self, geometry: ProbeGeometry) -> ProbeTransverseCoordinates:
        Y, X = numpy.mgrid[:geometry.heightInPixels, :geometry.widthInPixels]
        positionXInPixels = X - (geometry.widthInPixels - 1) / 2
        positionYInPixels = Y - (geometry.heightInPixels - 1) / 2

        positionXInMeters = positionXInPixels * geometry.pixelWidthInMeters
        positionYInMeters = positionYInPixels * geometry.pixelHeightInMeters

        return ProbeTransverseCoordinates(
            positionXInMeters=positionXInMeters,
            positionYInMeters=positionYInMeters,
        )

    def normalize(self, array: ProbeArrayType) -> ProbeArrayType:
        return array / numpy.sqrt(numpy.sum(numpy.abs(array)**2))

    def getName(self) -> str:
        return self._name.getValue()

    @abstractmethod
    def build(self) -> Probe:
        pass


class FromMemoryProbeBuilder(ProbeBuilder):

    def __init__(self, probe: Probe) -> None:
        super().__init__('From Memory')
        self._probe = probe

    def build(self) -> Probe:
        return self._probe


class FromFileProbeBuilder(ProbeBuilder):

    def __init__(self, filePath: Path, fileType: str, fileReader: ProbeFileReader) -> None:
        super().__init__('From File')
        self.filePath = self._registerPathParameter('FilePath', filePath)
        self.fileType = self._registerStringParameter('FileType', fileType)
        self._fileReader = fileReader

    def build(self) -> Probe:
        filePath = self.filePath.getValue()
        fileType = self.fileType.getValue()
        logger.debug(f'Reading \"{filePath}\" as \"{fileType}\"')

        try:
            probe = self._fileReader.read(filePath)
        except Exception as exc:
            raise RuntimeError(f'Failed to read \"{filePath}\"') from exc

        return probe

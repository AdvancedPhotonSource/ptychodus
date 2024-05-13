from __future__ import annotations
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
import logging

import numpy
import numpy.typing

from ptychodus.api.parametric import ParameterRepository
from ptychodus.api.probe import (Probe, ProbeFileReader, ProbeGeometry, ProbeGeometryProvider,
                                 WavefieldArrayType)
from ptychodus.api.typing import RealArrayType

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProbeTransverseCoordinates:
    positionXInMeters: RealArrayType
    positionYInMeters: RealArrayType

    @property
    def positionRInMeters(self) -> RealArrayType:
        return numpy.hypot(self.positionXInMeters, self.positionYInMeters)


class ProbeBuilder(ParameterRepository):

    def __init__(self, name: str) -> None:
        super().__init__('builder')
        self._name = self._registerStringParameter('name', name)

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

    def normalize(self, array: WavefieldArrayType) -> WavefieldArrayType:
        return array / numpy.sqrt(numpy.sum(numpy.abs(array)**2))

    def getName(self) -> str:
        return self._name.getValue()

    @abstractmethod
    def copy(self) -> ProbeBuilder:
        pass

    @abstractmethod
    def build(self, geometryProvider: ProbeGeometryProvider) -> Probe:
        pass


class FromMemoryProbeBuilder(ProbeBuilder):

    def __init__(self, probe: Probe) -> None:
        super().__init__('from_memory')
        self._probe = probe.copy()

    def copy(self) -> FromMemoryProbeBuilder:
        return FromMemoryProbeBuilder(self._probe)

    def build(self, geometryProvider: ProbeGeometryProvider) -> Probe:
        return self._probe


class FromFileProbeBuilder(ProbeBuilder):

    def __init__(self, filePath: Path, fileType: str, fileReader: ProbeFileReader) -> None:
        super().__init__('from_file')
        self.filePath = self._registerPathParameter('file_path', filePath)
        self.fileType = self._registerStringParameter('file_type', fileType)
        self._fileReader = fileReader

    def copy(self) -> FromFileProbeBuilder:
        return FromFileProbeBuilder(self.filePath.getValue(), self.fileType.getValue(),
                                    self._fileReader)

    def build(self, geometryProvider: ProbeGeometryProvider) -> Probe:
        filePath = self.filePath.getValue()
        fileType = self.fileType.getValue()
        logger.debug(f'Reading \"{filePath}\" as \"{fileType}\"')

        try:
            probe = self._fileReader.read(filePath)
        except Exception as exc:
            raise RuntimeError(f'Failed to read \"{filePath}\"') from exc

        return probe

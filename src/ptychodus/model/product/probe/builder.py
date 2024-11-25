from __future__ import annotations
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
import logging

import numpy
import numpy.typing

from ptychodus.api.parametric import ParameterGroup
from ptychodus.api.probe import (
    Probe,
    ProbeFileReader,
    ProbeGeometry,
    ProbeGeometryProvider,
    WavefieldArrayType,
)
from ptychodus.api.typing import RealArrayType

from .settings import ProbeSettings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProbeTransverseCoordinates:
    positionXInMeters: RealArrayType
    positionYInMeters: RealArrayType

    @property
    def positionRInMeters(self) -> RealArrayType:
        return numpy.hypot(self.positionXInMeters, self.positionYInMeters)


class ProbeBuilder(ParameterGroup):
    def __init__(self, settings: ProbeSettings, name: str) -> None:
        super().__init__()
        self._name = settings.builder.copy()
        self._name.setValue(name)
        self._addParameter('name', self._name)

    def getTransverseCoordinates(self, geometry: ProbeGeometry) -> ProbeTransverseCoordinates:
        Y, X = numpy.mgrid[: geometry.heightInPixels, : geometry.widthInPixels]
        positionXInPixels = X - (geometry.widthInPixels - 1) / 2
        positionYInPixels = Y - (geometry.heightInPixels - 1) / 2

        positionXInMeters = positionXInPixels * geometry.pixelWidthInMeters
        positionYInMeters = positionYInPixels * geometry.pixelHeightInMeters

        return ProbeTransverseCoordinates(
            positionXInMeters=positionXInMeters,
            positionYInMeters=positionYInMeters,
        )

    def normalize(self, array: WavefieldArrayType) -> WavefieldArrayType:
        return array / numpy.sqrt(numpy.sum(numpy.abs(array) ** 2))

    def getName(self) -> str:
        return self._name.getValue()

    def syncToSettings(self) -> None:
        for parameter in self.parameters().values():
            parameter.syncValueToParent()

    @abstractmethod
    def copy(self) -> ProbeBuilder:
        pass

    @abstractmethod
    def build(self, geometryProvider: ProbeGeometryProvider) -> Probe:
        pass


class FromMemoryProbeBuilder(ProbeBuilder):
    def __init__(self, settings: ProbeSettings, probe: Probe) -> None:
        super().__init__(settings, 'from_memory')
        self._settings = settings
        self._probe = probe.copy()

    def copy(self) -> FromMemoryProbeBuilder:
        return FromMemoryProbeBuilder(self._settings, self._probe)

    def build(self, geometryProvider: ProbeGeometryProvider) -> Probe:
        return self._probe


class FromFileProbeBuilder(ProbeBuilder):
    def __init__(
        self, settings: ProbeSettings, filePath: Path, fileType: str, fileReader: ProbeFileReader
    ) -> None:
        super().__init__(settings, 'from_file')
        self._settings = settings
        self.filePath = settings.filePath.copy()
        self.filePath.setValue(filePath)
        self._addParameter('file_path', self.filePath)
        self.fileType = settings.fileType.copy()
        self.fileType.setValue(fileType)
        self._addParameter('file_type', self.fileType)
        self._fileReader = fileReader

    def copy(self) -> FromFileProbeBuilder:
        return FromFileProbeBuilder(
            self._settings, self.filePath.getValue(), self.fileType.getValue(), self._fileReader
        )

    def build(self, geometryProvider: ProbeGeometryProvider) -> Probe:
        filePath = self.filePath.getValue()
        fileType = self.fileType.getValue()
        logger.debug(f'Reading "{filePath}" as "{fileType}"')

        try:
            probeFromFile = self._fileReader.read(filePath)
        except Exception as exc:
            raise RuntimeError(f'Failed to read "{filePath}"') from exc

        pixelGeometryFromFile = probeFromFile.getPixelGeometry()
        pixelGeometryFromProvider = geometryProvider.getProbeGeometry().getPixelGeometry()

        if pixelGeometryFromFile is None:
            return Probe(probeFromFile.getArray(), pixelGeometryFromProvider)

        # TODO remap probe from pixelGeometryFromFile to pixelGeometryFromProvider
        return probeFromFile

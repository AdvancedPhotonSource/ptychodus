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
        self._name.set_value(name)
        self._add_parameter('name', self._name)

    def getTransverseCoordinates(self, geometry: ProbeGeometry) -> ProbeTransverseCoordinates:
        Y, X = numpy.mgrid[: geometry.height_px, : geometry.width_px]
        positionXInPixels = X - (geometry.width_px - 1) / 2
        positionYInPixels = Y - (geometry.height_px - 1) / 2

        positionXInMeters = positionXInPixels * geometry.pixel_width_m
        positionYInMeters = positionYInPixels * geometry.pixel_height_m

        return ProbeTransverseCoordinates(
            positionXInMeters=positionXInMeters,
            positionYInMeters=positionYInMeters,
        )

    def normalize(self, array: WavefieldArrayType) -> WavefieldArrayType:
        return array / numpy.sqrt(numpy.sum(numpy.square(numpy.abs(array))))

    def getName(self) -> str:
        return self._name.get_value()

    def syncToSettings(self) -> None:
        for parameter in self.parameters().values():
            parameter.sync_value_to_parent()

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
        self.filePath.set_value(filePath)
        self._add_parameter('file_path', self.filePath)
        self.fileType = settings.fileType.copy()
        self.fileType.set_value(fileType)
        self._add_parameter('file_type', self.fileType)
        self._fileReader = fileReader

    def copy(self) -> FromFileProbeBuilder:
        return FromFileProbeBuilder(
            self._settings, self.filePath.get_value(), self.fileType.get_value(), self._fileReader
        )

    def build(self, geometryProvider: ProbeGeometryProvider) -> Probe:
        filePath = self.filePath.get_value()
        fileType = self.fileType.get_value()
        logger.debug(f'Reading "{filePath}" as "{fileType}"')

        try:
            probeFromFile = self._fileReader.read(filePath)
        except Exception as exc:
            raise RuntimeError(f'Failed to read "{filePath}"') from exc

        pixelGeometryFromFile = probeFromFile.get_pixel_geometry()
        pixelGeometryFromProvider = geometryProvider.get_probe_geometry().get_pixel_geometry()

        if pixelGeometryFromFile is None:
            return Probe(probeFromFile.get_array(), pixelGeometryFromProvider)

        # TODO remap probe from pixelGeometryFromFile to pixelGeometryFromProvider
        return probeFromFile

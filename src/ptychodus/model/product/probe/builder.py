from __future__ import annotations
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
import logging

import numpy
import numpy.typing

from ptychodus.api.parametric import ParameterGroup
from ptychodus.api.probe import (
    ProbeSequence,
    ProbeFileReader,
    ProbeGeometry,
    ProbeGeometryProvider,
    ComplexArrayType,
)
from ptychodus.api.typing import RealArrayType

from .settings import ProbeSettings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProbeTransverseCoordinates:
    position_x_m: RealArrayType
    position_y_m: RealArrayType

    @property
    def position_r_m(self) -> RealArrayType:
        return numpy.hypot(self.position_x_m, self.position_y_m)


class ProbeSequenceBuilder(ParameterGroup):
    def __init__(self, settings: ProbeSettings, name: str) -> None:
        super().__init__()
        self._name = settings.builder.copy()
        self._name.set_value(name)
        self._add_parameter('name', self._name)

    def get_transverse_coordinates(self, geometry: ProbeGeometry) -> ProbeTransverseCoordinates:
        Y, X = numpy.mgrid[: geometry.height_px, : geometry.width_px]  # noqa: N806
        position_x_px = X - (geometry.width_px - 1) / 2
        position_y_px = Y - (geometry.height_px - 1) / 2

        position_x_m = position_x_px * geometry.pixel_width_m
        position_y_m = position_y_px * geometry.pixel_height_m

        return ProbeTransverseCoordinates(
            position_x_m=position_x_m,
            position_y_m=position_y_m,
        )

    def normalize(self, array: ComplexArrayType) -> ComplexArrayType:
        return array / numpy.sqrt(numpy.sum(numpy.square(numpy.abs(array))))

    def get_name(self) -> str:
        return self._name.get_value()

    def sync_to_settings(self) -> None:
        for parameter in self.parameters().values():
            parameter.sync_value_to_parent()

    @abstractmethod
    def copy(self) -> ProbeSequenceBuilder:
        pass

    @abstractmethod
    def build(self, geometry_provider: ProbeGeometryProvider) -> ProbeSequence:
        pass


class FromMemoryProbeBuilder(ProbeSequenceBuilder):
    def __init__(self, settings: ProbeSettings, probe: ProbeSequence) -> None:
        super().__init__(settings, 'from_memory')
        self._settings = settings
        self._probe = probe.copy()

    def copy(self) -> FromMemoryProbeBuilder:
        return FromMemoryProbeBuilder(self._settings, self._probe)

    def build(self, geometry_provider: ProbeGeometryProvider) -> ProbeSequence:
        probe_geometry = geometry_provider.get_probe_geometry()

        try:
            pixel_geometry = self._probe.get_pixel_geometry()
        except ValueError:
            pixel_geometry = probe_geometry.get_pixel_geometry()

        try:
            opr_weights = self._probe.get_opr_weights()
        except ValueError:
            opr_weights = None

        # TODO regrid probe as needed based on probe geometry from file/provider
        return ProbeSequence(
            self._probe.get_array(),
            opr_weights,
            pixel_geometry,
        )


class FromFileProbeBuilder(ProbeSequenceBuilder):
    def __init__(
        self, settings: ProbeSettings, file_path: Path, file_type: str, file_reader: ProbeFileReader
    ) -> None:
        super().__init__(settings, 'from_file')
        self._settings = settings
        self.file_path = settings.file_path.copy()
        self.file_path.set_value(file_path)
        self._add_parameter('file_path', self.file_path)
        self.file_type = settings.file_type.copy()
        self.file_type.set_value(file_type)
        self._add_parameter('file_type', self.file_type)
        self._file_reader = file_reader

    def copy(self) -> FromFileProbeBuilder:
        return FromFileProbeBuilder(
            self._settings,
            self.file_path.get_value(),
            self.file_type.get_value(),
            self._file_reader,
        )

    def build(self, geometry_provider: ProbeGeometryProvider) -> ProbeSequence:
        file_path = self.file_path.get_value()
        file_type = self.file_type.get_value()
        logger.debug(f'Reading "{file_path}" as "{file_type}"')

        try:
            probe_from_file = self._file_reader.read(file_path)
        except Exception as exc:
            raise RuntimeError(f'Failed to read "{file_path}"') from exc

        probe_geometry = geometry_provider.get_probe_geometry()

        try:
            pixel_geometry = probe_from_file.get_pixel_geometry()
        except ValueError:
            pixel_geometry = probe_geometry.get_pixel_geometry()

        try:
            opr_weights = probe_from_file.get_opr_weights()
        except ValueError:
            opr_weights = None

        # TODO regrid probe as needed based on probe geometry from file/provider
        return ProbeSequence(
            probe_from_file.get_array(),
            opr_weights,
            pixel_geometry,
        )

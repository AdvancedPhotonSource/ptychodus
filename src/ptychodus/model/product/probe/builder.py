from __future__ import annotations
from abc import abstractmethod
from collections.abc import Sequence
from enum import auto, IntEnum
import logging

import numpy

from ptychodus.api.parametric import ParameterGroup
from ptychodus.api.probe_gen import generate_coherent_probe_modes, generate_incoherent_probe_modes
from ptychodus.api.probe import (
    Probe,
    ProbeSequence,
    ProbeFileReader,
    ProbeGeometryProvider,
)

from .settings import ProbeSettings

logger = logging.getLogger(__name__)


class ProbeModeDecayType(IntEnum):
    NONE = auto()
    POLYNOMIAL = auto()
    EXPONENTIAL = auto()

    def get_weights(self, num_modes: int, decay_ratio: float) -> Sequence[float]:
        match self.value:
            case ProbeModeDecayType.EXPONENTIAL:
                b = 1.0 / decay_ratio
                return [b**-n for n in range(num_modes)]
            case ProbeModeDecayType.POLYNOMIAL:
                b = numpy.log(decay_ratio) / numpy.log(2.0)
                return [(n + 1) ** b for n in range(num_modes)]
            case _:
                return [1.0] + [0.0] * (num_modes - 1)


class ProbeSequenceBuilder(ParameterGroup):
    def __init__(self, settings: ProbeSettings, name: str) -> None:
        super().__init__()
        self._name = settings.builder.copy()
        self._name.set_value(name)
        self._add_parameter('name', self._name)

        self.num_incoherent_modes = settings.num_incoherent_modes.copy()
        self._add_parameter('num_incoherent_modes', self.num_incoherent_modes)

        self.orthogonalize_incoherent_modes = settings.orthogonalize_incoherent_modes.copy()
        self._add_parameter('orthogonalize_incoherent_modes', self.orthogonalize_incoherent_modes)

        self.incoherent_mode_decay_type = settings.incoherent_mode_decay_type.copy()
        self._add_parameter('incoherent_mode_decay_type', self.incoherent_mode_decay_type)

        self.incoherent_mode_decay_ratio = settings.incoherent_mode_decay_ratio.copy()
        self._add_parameter('incoherent_mode_decay_ratio', self.incoherent_mode_decay_ratio)

        self.num_coherent_modes = settings.num_coherent_modes.copy()
        self._add_parameter('num_coherent_modes', self.num_coherent_modes)

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

    def _get_imode_weights(self) -> Sequence[float]:
        imode_decay_ratio = self.incoherent_mode_decay_ratio.get_value()
        imode_decay_type_text = self.incoherent_mode_decay_type.get_value()
        imode_decay_type = ProbeModeDecayType.NONE

        if imode_decay_ratio > 0.0:
            try:
                imode_decay_type = ProbeModeDecayType[imode_decay_type_text.upper()]
            except KeyError:
                logger.debug(f'Unknown probe mode decay type "{imode_decay_type_text}"')

        num_imodes = self.num_incoherent_modes.get_value()
        return imode_decay_type.get_weights(num_imodes, imode_decay_ratio)

    def _build_probe_modes(
        self, rng: numpy.random.Generator, probe: Probe, num_diffraction_patterns: int
    ) -> ProbeSequence:
        probe_with_imodes = generate_incoherent_probe_modes(
            rng,
            probe,
            self._get_imode_weights(),
            orthogonalize=self.orthogonalize_incoherent_modes.get_value(),
        )
        probe_seq = generate_coherent_probe_modes(
            rng,
            probe_with_imodes,
            num_cmodes=self.num_coherent_modes.get_value(),
            num_diffraction_patterns=num_diffraction_patterns,
        )
        array = probe_seq.get_array()
        logger.debug(f'Multimodal probe {array.shape=}')
        return probe_seq


class FromMemoryProbeBuilder(ProbeSequenceBuilder):
    def __init__(self, settings: ProbeSettings, probe: ProbeSequence) -> None:
        super().__init__(settings, 'from_memory')
        self._settings = settings
        self._probe = probe.copy()

    def copy(self) -> FromMemoryProbeBuilder:
        builder = FromMemoryProbeBuilder(self._settings, self._probe)

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        return builder

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
    def __init__(self, settings: ProbeSettings, file_reader: ProbeFileReader) -> None:
        super().__init__(settings, 'from_file')
        self._settings = settings
        self._file_reader = file_reader

        self.file_path = settings.file_path.copy()
        self._add_parameter('file_path', self.file_path)

        self.file_type = settings.file_type.copy()
        self._add_parameter('file_type', self.file_type)

    def copy(self) -> FromFileProbeBuilder:
        builder = FromFileProbeBuilder(self._settings, self._file_reader)

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        return builder

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

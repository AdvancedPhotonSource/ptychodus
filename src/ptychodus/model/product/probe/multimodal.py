from __future__ import annotations
from collections.abc import Sequence
from enum import auto, IntEnum
import logging

import numpy.random
import scipy.linalg

from ptychodus.api.parametric import ParameterGroup
from ptychodus.api.probe import ProbeSequence, ProbeGeometryProvider
from ptychodus.api.propagator import intensity
from ptychodus.api.typing import ComplexArrayType, RealArrayType

from .settings import ProbeSettings

logger = logging.getLogger(__name__)


class ProbeModeDecayType(IntEnum):
    NONE = auto()
    POLYNOMIAL = auto()
    EXPONENTIAL = auto()

    def get_weights(self, num_modes: int, decay_ratio: float) -> Sequence[float]:
        match self.value:
            case ProbeModeDecayType.EXPONENTIAL:
                b = 1.0 + (1.0 - decay_ratio) / decay_ratio
                return [b**-n for n in range(num_modes)]
            case ProbeModeDecayType.POLYNOMIAL:
                b = numpy.log(decay_ratio) / numpy.log(2.0)
                return [(n + 1) ** b for n in range(num_modes)]
            case _:
                return [1.0] + [0.0] * (num_modes - 1)


class MultimodalProbeBuilder(ParameterGroup):
    def __init__(self, rng: numpy.random.Generator, settings: ProbeSettings) -> None:
        super().__init__()
        self._rng = rng
        self._settings = settings

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

    def sync_to_settings(self) -> None:
        for parameter in self.parameters().values():
            parameter.sync_value_to_parent()

    def copy(self) -> MultimodalProbeBuilder:
        builder = MultimodalProbeBuilder(self._rng, self._settings)

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        return builder

    def _orthogonalize_incoherent_modes(self, array_in: ComplexArrayType) -> ComplexArrayType:
        array_out = array_in.copy()

        if array_in.shape[-3] > 1:
            imodes_as_rows = array_in[0].reshape(array_in.shape[-3], -1)
            imodes_as_cols = imodes_as_rows.T

            try:
                imodes_as_ortho_cols = scipy.linalg.orth(imodes_as_cols)
            except ValueError as ex:
                logger.exception(ex)
                return array_in.copy()

            imodes_as_ortho_rows = imodes_as_ortho_cols.T
            imodes_ortho = imodes_as_ortho_rows.reshape(*array_in.shape)

            array_out[0, :, :, :] = imodes_ortho

        return array_out

    def _get_imode_weights(self, num_imodes: int) -> Sequence[float]:
        imode_decay_type_text = self.incoherent_mode_decay_type.get_value()
        imode_decay_ratio = self.incoherent_mode_decay_ratio.get_value()
        imode_decay_type = ProbeModeDecayType.NONE

        if imode_decay_ratio > 0.0:
            try:
                imode_decay_type = ProbeModeDecayType[imode_decay_type_text.upper()]
            except KeyError:
                logger.debug(f'Unknown probe mode decay type "{imode_decay_type_text}"')

        return imode_decay_type.get_weights(num_imodes, imode_decay_ratio)

    def _adjust_imode_power(self, array_in: ComplexArrayType, power: float) -> ComplexArrayType:
        imode_weights = self._get_imode_weights(array_in.shape[-3])
        array_out = array_in.copy()
        it = iter(array_out[0])  # iterate incoherent modes

        for weight in imode_weights:
            imode = next(it)
            ipower = numpy.sum(numpy.square(numpy.abs(imode)))
            imode *= numpy.sqrt(weight * power / ipower)

        return array_out

    def _random_phase_shift_axis(self, size: int) -> ComplexArrayType:
        a = self._rng.uniform() - 0.5
        b = (size - 1 - 2 * numpy.arange(size)) / size
        return numpy.exp(1j * numpy.pi * a * b)

    def _init_modes(
        self,
        geometry_provider: ProbeGeometryProvider,
        array_in: ComplexArrayType,
        normalize_cmodes: bool = True,
    ) -> ComplexArrayType:
        assert array_in.ndim == 4
        num_cmodes = self.num_coherent_modes.get_value()
        num_imodes = self.num_incoherent_modes.get_value()
        height = array_in.shape[-2]
        width = array_in.shape[-1]

        array_out = numpy.zeros((num_cmodes, num_imodes, height, width), array_in.dtype)

        for cmode in range(num_cmodes):
            if cmode < array_in.shape[0]:
                # copy existing cmode
                values = array_in[cmode, 0, :, :]
            else:
                # randomize new cmode
                real = self._rng.normal(0.0, 1.0, size=(height, width))
                imag = self._rng.normal(0.0, 1.0, size=(height, width))
                values = real + 1j * imag

                if normalize_cmodes:
                    values /= numpy.sqrt(numpy.mean(intensity(values)))

            array_out[cmode, 0, :, :] = values

        for imode in range(num_imodes):
            if imode < array_in.shape[1]:
                # copy existing imode
                values = array_in[0, imode, :, :]
            else:
                # apply random phase shift to first imode
                first_imode = array_in[0, 0, :, :]
                phase_shift_y = self._random_phase_shift_axis(height)
                phase_shift_x = self._random_phase_shift_axis(width)
                values = first_imode * numpy.outer(phase_shift_y, phase_shift_x)

            array_out[0, imode, :, :] = values

        if self.orthogonalize_incoherent_modes.get_value():
            array_out = self._orthogonalize_incoherent_modes(array_out)

        if geometry_provider.probe_photon_count > 0.0:
            array_out = self._adjust_imode_power(array_out, geometry_provider.probe_photon_count)

        return array_out

    def _init_opr_weights(
        self, geometry_provider: ProbeGeometryProvider, small_value: float = 1.0e-6
    ) -> RealArrayType | None:
        num_scan_points = geometry_provider.num_scan_points
        num_cmodes = self.num_coherent_modes.get_value()
        opr_weights: RealArrayType | None = None

        if self.num_coherent_modes.get_value() > 1:
            opr_weights = small_value * self._rng.normal(
                0.0, 1.0, size=(num_scan_points, num_cmodes)
            )
            assert opr_weights is not None  # unnecessary but makes pylance less annoying
            opr_weights[:, 0] = 1.0

        return opr_weights

    def set_identity(self) -> None:
        self.num_coherent_modes.set_value(1)
        self.num_incoherent_modes.set_value(1)

    def build(
        self, probes: ProbeSequence, geometry_provider: ProbeGeometryProvider
    ) -> ProbeSequence:
        if self.num_coherent_modes.get_value() <= 1 and self.num_incoherent_modes.get_value() <= 1:
            return probes

        array = self._init_modes(geometry_provider, probes.get_array())

        try:
            opr_weights: RealArrayType | None = probes.get_opr_weights()
        except ValueError:
            opr_weights = self._init_opr_weights(geometry_provider)
        else:
            # TODO if opr_weights.shape[0] != num_scan_points: raise ValueError()
            pass

        return ProbeSequence(
            array=array,
            opr_weights=opr_weights,
            pixel_geometry=probes.get_pixel_geometry(),
        )

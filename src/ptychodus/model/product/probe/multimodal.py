from __future__ import annotations
from collections.abc import Sequence
from enum import auto, IntEnum
import logging

import numpy
import scipy.linalg

from ptychodus.api.parametric import ParameterGroup
from ptychodus.api.probe import Probe, ProbeGeometryProvider, WavefieldArrayType

from .settings import ProbeSettings

logger = logging.getLogger(__name__)


class ProbeModeDecayType(IntEnum):
    POLYNOMIAL = auto()
    EXPONENTIAL = auto()


class MultimodalProbeBuilder(ParameterGroup):
    def __init__(self, rng: numpy.random.Generator, settings: ProbeSettings) -> None:
        super().__init__()
        self._rng = rng
        self._settings = settings

        self.numberOfIncoherentModes = settings.numberOfIncoherentModes.copy()
        self._add_parameter('number_of_incoherent_modes', self.numberOfIncoherentModes)

        self.incoherentModeDecayType = settings.incoherentModeDecayType.copy()
        self._add_parameter('incoherent_mode_decay_type', self.incoherentModeDecayType)

        self.incoherentModeDecayRatio = settings.incoherentModeDecayRatio.copy()
        self._add_parameter('incoherent_mode_decay_ratio', self.incoherentModeDecayRatio)

        self.orthogonalizeIncoherentModes = settings.orthogonalizeIncoherentModes.copy()
        self._add_parameter('orthogonalize_incoherent_modes', self.orthogonalizeIncoherentModes)

    def syncToSettings(self) -> None:
        for parameter in self.parameters().values():
            parameter.sync_value_to_parent()

    def copy(self) -> MultimodalProbeBuilder:
        builder = MultimodalProbeBuilder(self._rng, self._settings)

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        return builder

    def _init_modes(self, probe: WavefieldArrayType) -> WavefieldArrayType:
        # TODO OPR
        assert probe.ndim == 4
        array = numpy.tile(probe[0, 0, :, :], (self.numberOfIncoherentModes.get_value(), 1, 1))
        it = iter(array)  # iterate incoherent modes
        next(it)  # preserve the first incoherent mode

        for imode in it:  # phase shift the rest
            pw = probe.shape[-1]  # TODO clean up
            variate1 = self._rng.uniform(size=(2, 1)) - 0.5
            variate2 = (numpy.arange(pw) + 0.5) / pw - 0.5
            ps = numpy.exp(-2j * numpy.pi * variate1 * variate2)
            imode *= ps[0][numpy.newaxis] * ps[1][:, numpy.newaxis]

        return array

    def _orthogonalizeIncoherentModes(self, probe: WavefieldArrayType) -> WavefieldArrayType:
        # TODO OPR
        imodes_as_rows = probe.reshape(probe.shape[-3], -1)
        imodes_as_cols = imodes_as_rows.T
        imodes_as_ortho_cols = scipy.linalg.orth(imodes_as_cols)
        imodes_as_ortho_rows = imodes_as_ortho_cols.T
        return imodes_as_ortho_rows.reshape(*probe.shape)

    def _get_imode_weights(self, num_imodes: int) -> Sequence[float]:
        imode_decay_type_text = self.incoherentModeDecayType.get_value()
        imode_decay_ratio = self.incoherentModeDecayRatio.get_value()

        if imode_decay_ratio > 0.0:
            try:
                imode_decay_type = ProbeModeDecayType[imode_decay_type_text.upper()]
            except KeyError:
                imode_decay_type = ProbeModeDecayType.POLYNOMIAL

            if imode_decay_type == ProbeModeDecayType.EXPONENTIAL:
                b = 1.0 + (1.0 - imode_decay_ratio) / imode_decay_ratio
                return [b**-n for n in range(num_imodes)]
            else:
                b = numpy.log(imode_decay_ratio) / numpy.log(2.0)
                return [(n + 1) ** b for n in range(num_imodes)]

        return [1.0] + [0.0] * (num_imodes - 1)

    def _adjust_power(self, probe: WavefieldArrayType, power: float) -> WavefieldArrayType:
        imode_weights = self._get_imode_weights(probe.shape[-3])
        array = probe.copy()
        it = iter(array)  # iterate incoherent modes

        for weight in imode_weights:
            imode = next(it)
            ipower = numpy.sum(numpy.square(numpy.abs(imode)))
            imode *= numpy.sqrt(weight * power / ipower)

        return array

    def build(self, probe: Probe, geometryProvider: ProbeGeometryProvider) -> Probe:
        array = self._init_modes(probe.get_array())

        if self.orthogonalizeIncoherentModes.get_value():
            array = self._orthogonalizeIncoherentModes(array)

        power = probe.get_intensity().sum()

        if geometryProvider.probe_photon_count > 0.0:
            power = geometryProvider.probe_photon_count

        return Probe(self._adjust_power(array, power), probe.get_pixel_geometry())

from __future__ import annotations
from collections.abc import Sequence
from enum import auto, IntEnum
import logging

import numpy
import scipy.linalg

from ptychodus.api.parametric import ParameterRepository
from ptychodus.api.probe import Probe, ProbeArrayType

logger = logging.getLogger(__name__)


class ProbeModeDecayType(IntEnum):
    POLYNOMIAL = auto()
    EXPONENTIAL = auto()


class MultimodalProbeBuilder(ParameterRepository):

    def __init__(self, rng: numpy.random.Generator) -> None:
        super().__init__('Additional Modes')
        self._rng = rng

        self.isOrthogonalizeModesEnabled = self._registerBooleanParameter(
            'IsOrthogonalizeModesEnabled', True)
        self.numberOfModes = self._registerIntegerParameter('NumberOfModes', 1, minimum=1)
        self.modeDecayType = self._registerStringParameter('ProbeModeDecayType', 'Polynomial')
        self.modeDecayRatio = self._registerRealParameter('ModeDecayRatio',
                                                          1.,
                                                          minimum=0.,
                                                          maximum=1.)

    def copy(self) -> MultimodalProbeBuilder:
        builder = MultimodalProbeBuilder(self._rng)
        builder.isOrthogonalizeModesEnabled.setValue(self.isOrthogonalizeModesEnabled.getValue())
        builder.numberOfModes.setValue(self.numberOfModes.getValue())
        builder.modeDecayType.setValue(self.modeDecayType.getValue())
        builder.modeDecayRatio.setValue(self.modeDecayRatio.getValue())
        return builder

    def _initializeModes(self, probe: ProbeArrayType) -> ProbeArrayType:
        modeList: list[ProbeArrayType] = list()

        if probe.ndim == 2:
            modeList.append(probe)
        elif probe.ndim >= 3:
            probe3D = probe

            while probe3D.ndim > 3:
                probe3D = probe3D[0]

            for mode in probe3D:
                modeList.append(mode)
        else:
            raise ValueError('Probe array must contain at least two dimensions.')

        for mode in range(self.numberOfModes.getValue() - 1):
            # randomly shift the first mode
            pw = probe.shape[-1]  # TODO clean up
            variate1 = self._rng.uniform(size=(2, 1)) - 0.5
            variate2 = (numpy.arange(0, pw) + 0.5) / pw - 0.5
            ps = numpy.exp(-2j * numpy.pi * variate1 * variate2)
            phaseShift = ps[0][numpy.newaxis] * ps[1][:, numpy.newaxis]
            mode = modeList[0] * phaseShift
            modeList.append(mode)

        return numpy.stack(modeList)

    def _orthogonalizeModes(self, probe: ProbeArrayType) -> ProbeArrayType:
        probeModesAsRows = probe.reshape(probe.shape[-3], -1)
        probeModesAsCols = probeModesAsRows.T
        probeModesAsOrthoCols = scipy.linalg.orth(probeModesAsCols)
        probeModesAsOrthoRows = probeModesAsOrthoCols.T
        return probeModesAsOrthoRows.reshape(*probe.shape)

    def _getModeWeights(self, totalNumberOfModes: int) -> Sequence[float]:
        modeDecayTypeText = self.modeDecayType.getValue()
        modeDecayRatio = self.modeDecayRatio.getValue()

        if modeDecayRatio > 0.:
            try:
                modeDecayType = ProbeModeDecayType[modeDecayTypeText.upper()]
            except KeyError:
                modeDecayType = ProbeModeDecayType.POLYNOMIAL

            if modeDecayType == ProbeModeDecayType.EXPONENTIAL:
                b = 1. + (1. - modeDecayRatio) / modeDecayRatio
                return [b**-n for n in range(totalNumberOfModes)]
            else:
                b = numpy.log(modeDecayRatio) / numpy.log(2.)
                return [(n + 1)**b for n in range(totalNumberOfModes)]

        return [1.] + [0.] * (totalNumberOfModes - 1)

    def _adjustRelativePower(self, probe: ProbeArrayType) -> ProbeArrayType:
        modeWeights = self._getModeWeights(probe.shape[-3])
        power0 = numpy.sum(numpy.square(numpy.abs(probe[0, ...])))
        adjustedProbe = probe.copy()

        for modeIndex, weight in enumerate(modeWeights):
            powerN = numpy.sum(numpy.square(numpy.abs(adjustedProbe[modeIndex, ...])))
            adjustedProbe[modeIndex, ...] *= numpy.sqrt(weight * power0 / powerN)

        return adjustedProbe

    def build(self, probe: Probe) -> Probe:
        if self.numberOfModes.getValue() <= 1:
            return probe

        array = self._initializeModes(probe.array)

        if self.isOrthogonalizeModesEnabled.getValue():
            array = self._orthogonalizeModes(array)

        array = self._adjustRelativePower(array)

        return Probe(array,
                     pixelWidthInMeters=probe.pixelWidthInMeters,
                     pixelHeightInMeters=probe.pixelHeightInMeters)

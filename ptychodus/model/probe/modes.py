from __future__ import annotations
from collections.abc import Sequence
from decimal import Decimal
from enum import auto, Enum
import logging

import numpy
import scipy.linalg

from ...api.probe import ProbeArrayType

logger = logging.getLogger(__name__)


class ProbeModeDecayType(Enum):
    POLYNOMIAL = auto()
    EXPONENTIAL = auto()


class MultimodalProbeFactory:

    def __init__(self, rng: numpy.random.Generator) -> None:
        super().__init__()
        self._rng = rng

    def _initializeModes(self, probe: ProbeArrayType, numberOfModes: int) -> ProbeArrayType:
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

        while len(modeList) < numberOfModes:
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

    def _getModeWeights(self, numberOfModes: int, modeDecayType: ProbeModeDecayType,
                        modeDecayRatio: Decimal) -> Sequence[float]:
        weights = [1.] * numberOfModes

        if Decimal(0) < modeDecayRatio and modeDecayRatio < Decimal(1):
            if modeDecayType == ProbeModeDecayType.EXPONENTIAL:
                b = float(1 + (1 - modeDecayRatio) / modeDecayRatio)
                weights = [b**-n for n in range(numberOfModes)]
            else:
                b = float(modeDecayRatio.ln() / Decimal(2).ln())
                weights = [(n + 1)**b for n in range(numberOfModes)]

        return weights

    def _adjustRelativePower(self, probe: ProbeArrayType, numberOfModes: int,
                             modeDecayType: ProbeModeDecayType,
                             modeDecayRatio: Decimal) -> ProbeArrayType:
        modeWeights = self._getModeWeights(probe.shape[-3], modeDecayType, modeDecayRatio)
        power0 = numpy.sum(numpy.square(numpy.abs(probe[0, ...])))
        adjustedProbe = probe.copy()

        for modeIndex, weight in enumerate(modeWeights):
            powerN = numpy.sum(numpy.square(numpy.abs(adjustedProbe[modeIndex, ...])))
            adjustedProbe[modeIndex, ...] *= numpy.sqrt(weight * power0 / powerN)

        return adjustedProbe

    def build(self, initialProbe: ProbeArrayType, numberOfModes: int,
              orthogonalizeModesEnabled: bool, modeDecayType: ProbeModeDecayType,
              modeDecayRatio: Decimal) -> ProbeArrayType:
        probe = self._initializeModes(initialProbe, numberOfModes)

        if orthogonalizeModesEnabled:
            probe = self._orthogonalizeModes(probe)

        return self._adjustRelativePower(probe, numberOfModes, modeDecayType, modeDecayRatio)

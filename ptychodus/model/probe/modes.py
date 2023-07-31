from __future__ import annotations
from collections.abc import Sequence
from decimal import Decimal
from enum import auto, Enum
from typing import Final
import logging

import numpy
import scipy.linalg

from ...api.geometry import Interval
from ...api.observer import Observable
from ...api.probe import ProbeArrayType
from .settings import ProbeSettings

logger = logging.getLogger(__name__)


class ProbeModeDecayType(Enum):
    POLYNOMIAL = auto()
    EXPONENTIAL = auto()


class MultimodalProbeFactory(Observable):
    MAX_INT: Final[int] = 0x7FFFFFFF

    def __init__(self, rng: numpy.random.Generator) -> None:
        super().__init__()
        self._rng = rng
        self._numberOfModes = 0
        self._orthogonalizeModesEnabled = False
        self._modeDecayType = ProbeModeDecayType.POLYNOMIAL
        self._modeDecayRatio = Decimal(1)

    def getNumberOfModesLimits(self) -> Interval[int]:
        return Interval[int](1, self.MAX_INT)

    def getNumberOfModes(self) -> int:
        limits = self.getNumberOfModesLimits()
        return limits.clamp(self._numberOfModes)

    def setNumberOfModes(self, number: int) -> None:
        if self._numberOfModes != number:
            self._numberOfModes = number
            self.notifyObservers()

    @property
    def isOrthogonalizeModesEnabled(self) -> bool:
        return self._orthogonalizeModesEnabled

    def setOrthogonalizeModesEnabled(self, value: bool) -> None:
        if self._orthogonalizeModesEnabled != value:
            self._orthogonalizeModesEnabled = value
            self.notifyObservers()

    def getModeDecayType(self) -> ProbeModeDecayType:
        return self._modeDecayType

    def setModeDecayType(self, value: ProbeModeDecayType) -> None:
        if self._modeDecayType != value:
            self._modeDecayType = value
            self.notifyObservers()

    def getModeDecayRatioLimits(self) -> Interval[Decimal]:
        return Interval[Decimal](Decimal(0), Decimal(1))

    def getModeDecayRatio(self) -> Decimal:
        limits = self.getModeDecayRatioLimits()
        return limits.clamp(self._modeDecayRatio)

    def setModeDecayRatio(self, value: Decimal) -> None:
        if self._modeDecayRatio != value:
            self._modeDecayRatio = value
            self.notifyObservers()

    def syncFromSettings(self, settings: ProbeSettings) -> None:
        self._numberOfModes = settings.numberOfModes.value
        self._orthogonalizeModesEnabled = settings.orthogonalizeModesEnabled.value

        try:
            self._modeDecayType = ProbeModeDecayType[settings.modeDecayType.value.upper()]
        except KeyError:
            self._modeDecayType = ProbeModeDecayType.POLYNOMIAL

        self._modeDecayRatio = settings.modeDecayRatio.value
        self.notifyObservers()

    def syncToSettings(self, settings: ProbeSettings) -> None:
        settings.numberOfModes.value = self._numberOfModes
        settings.orthogonalizeModesEnabled.value = self._orthogonalizeModesEnabled
        settings.modeDecayType.value = self._modeDecayType.name
        settings.modeDecayRatio.value = self._modeDecayRatio

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

        while len(modeList) < self.getNumberOfModes():
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

    def _getModeWeights(self, numberOfModes: int) -> Sequence[float]:
        weights = [1.] * numberOfModes
        decayRatio = self.getModeDecayRatio()
        decayRatioLimits = self.getModeDecayRatioLimits()

        if decayRatio in decayRatioLimits:
            if self._modeDecayType == ProbeModeDecayType.EXPONENTIAL.value:
                b = float(1 + (1 - decayRatio) / decayRatio)
                weights = [b**-n for n in range(numberOfModes)]
            else:
                b = float(decayRatio.ln() / Decimal(2).ln())
                weights = [(n + 1)**b for n in range(numberOfModes)]

        return weights

    def _adjustRelativePower(self, probe: ProbeArrayType) -> ProbeArrayType:
        modeWeights = self._getModeWeights(probe.shape[-3])
        power0 = numpy.sum(numpy.square(numpy.abs(probe[0, ...])))
        adjustedProbe = probe.copy()

        for modeIndex, weight in enumerate(modeWeights):
            powerN = numpy.sum(numpy.square(numpy.abs(adjustedProbe[modeIndex, ...])))
            adjustedProbe[modeIndex, ...] *= numpy.sqrt(weight * power0 / powerN)

        return adjustedProbe

    def build(self, initialProbe: ProbeArrayType) -> ProbeArrayType:
        probe = self._initializeModes(initialProbe)

        if self._orthogonalizeModesEnabled:
            probe = self._orthogonalizeModes(probe)

        return self._adjustRelativePower(probe)

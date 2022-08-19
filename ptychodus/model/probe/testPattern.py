from __future__ import annotations
import numpy

from ...api.probe import ProbeArrayType
from .initializer import ProbeInitializer
from .settings import ProbeSettings
from .sizer import ProbeSizer


class TestPatternProbeInitializer(ProbeInitializer):

    def __init__(self, sizer: ProbeSizer) -> None:
        super().__init__()
        self._sizer = sizer

    @classmethod
    def createInstance(cls, sizer: ProbeSizer) -> TestPatternProbeInitializer:
        return cls(sizer)

    def syncFromSettings(self, settings: ProbeSettings) -> None:
        super().syncFromSettings(settings)

    def syncToSettings(self, settings: ProbeSettings) -> None:
        super().syncToSettings(settings)

    @property
    def displayName(self) -> str:
        return 'Test Pattern'

    @property
    def simpleName(self) -> str:
        return super().simpleName

    def __call__(self) -> ProbeArrayType:
        probeSize_px = self._sizer.getProbeSize()
        gridCenter_px = probeSize_px / 2
        cellCenters = (numpy.arange(probeSize_px) + 0.5 - gridCenter_px) / gridCenter_px

        Y, X = numpy.meshgrid(cellCenters, cellCenters)
        NIL = (numpy.hypot(X, Y) > 1)
        X[NIL] = 0
        Y[NIL] = 0

        return X + 1j * Y

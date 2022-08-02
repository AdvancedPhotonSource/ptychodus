import numpy

from ...api.probe import ProbeArrayType
from .settings import ProbeSettings


class TestPatternProbeInitializer:

    def __init__(self, settings: ProbeSettings) -> None:
        self._settings = settings

    def __call__(self) -> ProbeArrayType:
        probeSize_px = self._settings.probeSize.value
        gridCenter_px = probeSize_px / 2
        cellCenters = (numpy.arange(probeSize_px) + 0.5 - gridCenter_px) / gridCenter_px

        Y, X = numpy.meshgrid(cellCenters, cellCenters)
        NIL = (numpy.hypot(X, Y) > 1)
        X[NIL] = 0
        Y[NIL] = 0

        return X + 1j * Y

import numpy

from ...api.probe import ProbeArrayType
from .settings import ProbeSettings


class TestPatternProbeInitializer:

    def __init__(self, settings: ProbeSettings) -> None:
        self._settings = settings

    def __call__(self) -> ProbeArrayType:
        hsize_px = self._settings.probeSize.value / 2

        Y, X = (numpy.mgrid[-hsize_px:hsize_px, -hsize_px:hsize_px] + 0.5) / hsize_px
        NIL = (numpy.hypot(X, Y) > 1)
        X[NIL] = 0
        Y[NIL] = 0

        return X + 1j * Y

from __future__ import annotations
from typing import Final

import numpy

from ...api.probe import ProbeArrayType
from .repository import ProbeInitializer
from .settings import ProbeSettings
from .sizer import ProbeSizer


class DiskProbeInitializer(ProbeInitializer):
    SIMPLE_NAME: Final[str] = 'Disk'
    DISPLAY_NAME: Final[str] = 'Disk'

    def __init__(self, sizer: ProbeSizer) -> None:
        super().__init__()
        self._sizer = sizer

    @property
    def simpleName(self) -> str:
        return self.SIMPLE_NAME

    @property
    def displayName(self) -> str:
        return self.DISPLAY_NAME

    def syncFromSettings(self, settings: ProbeSettings) -> None:
        pass

    def syncToSettings(self, settings: ProbeSettings) -> None:
        pass

    def __call__(self) -> ProbeArrayType:
        extent = self._sizer.getProbeExtent()

        gridCenterX = extent.width / 2
        cellCentersX = (numpy.arange(extent.width) + 0.5 - gridCenterX) / gridCenterX

        gridCenterY = extent.height / 2
        cellCentersY = (numpy.arange(extent.height) + 0.5 - gridCenterY) / gridCenterY

        Y, X = numpy.meshgrid(cellCentersY, cellCentersX)
        gridCentersR = numpy.hypot(X, Y)

        # FIXME regular disk vs test pattern
        NIL = (gridCentersR > 1)
        X[NIL] = 0
        Y[NIL] = 0

        return X + 1j * Y

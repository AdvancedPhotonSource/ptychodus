from __future__ import annotations

from ...api.geometry import Interval
from ...api.image import ImageExtent
from ...api.observer import Observable, Observer
from ..data import DiffractionPatternSizer
from .settings import ProbeSettings


class ProbeSizer(Observable, Observer):

    def __init__(self, settings: ProbeSettings,
                 diffractionPatternSizer: DiffractionPatternSizer) -> None:
        super().__init__()
        self._settings = settings
        self._diffractionPatternSizer = diffractionPatternSizer

    @classmethod
    def createInstance(cls, settings: ProbeSettings,
                       diffractionPatternSizer: DiffractionPatternSizer) -> ProbeSizer:
        sizer = cls(settings, diffractionPatternSizer)
        settings.addObserver(sizer)
        diffractionPatternSizer.addObserver(sizer)
        return sizer

    @property
    def _probeSizeMax(self) -> int:
        cropX = self._diffractionPatternSizer.getExtentXInPixels()
        cropY = self._diffractionPatternSizer.getExtentYInPixels()
        return min(cropX, cropY)

    def getProbeSizeLimits(self) -> Interval[int]:
        return Interval[int](1, self._probeSizeMax)

    def getProbeSize(self) -> int:
        limits = self.getProbeSizeLimits()
        return limits.clamp(self._settings.probeSize.value)

    def getProbeExtent(self) -> ImageExtent:
        size = self.getProbeSize()
        return ImageExtent(width=size, height=size)

    def _updateProbeSize(self) -> None:
        if self._settings.automaticProbeSizeEnabled.value:
            self._settings.probeSize.value = self._probeSizeMax

        self.notifyObservers()

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self._updateProbeSize()
        elif observable is self._diffractionPatternSizer:
            self._updateProbeSize()

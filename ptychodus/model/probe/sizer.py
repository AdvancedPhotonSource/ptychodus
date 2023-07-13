from __future__ import annotations

from ...api.image import ImageExtent
from ...api.observer import Observable, Observer
from ..data import DiffractionPatternSizer


class ProbeSizer(Observable, Observer):

    def __init__(self, diffractionPatternSizer: DiffractionPatternSizer) -> None:
        super().__init__()
        self._diffractionPatternSizer = diffractionPatternSizer

    @classmethod
    def createInstance(cls, diffractionPatternSizer: DiffractionPatternSizer) -> ProbeSizer:
        sizer = cls(diffractionPatternSizer)
        diffractionPatternSizer.addObserver(sizer)
        return sizer

    def getExtentInPixels(self) -> ImageExtent:
        return self._diffractionPatternSizer.getExtentInPixels()

    def update(self, observable: Observable) -> None:
        if observable is self._diffractionPatternSizer:
            self.notifyObservers()

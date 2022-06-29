from __future__ import annotations
from decimal import Decimal

import numpy

from ..api.scan import ScanPoint, ScanPointSequence, ScanPointTransform


class TransformedScanPointSequence(ScanPointSequence, Observable, Observer):

    @staticmethod
    def _createTransformEntry(xform: ScanPointTransform) -> PluginEntry[ScanPointTransform]:
        return PluginEntry[ScanPointTransform](simpleName=xform.simpleName,
                                               displayName=xform.displayName,
                                               strategy=xform)

    def __init__(self, rng: numpy.random.Generator, settings: ScanSettings,
                 scanPointSequence: ScanPointSequence) -> None:
        super().__init__()
        self._rng = rng
        self._settings = settings
        self._scanPointSequence = scanPointSequence
        self._transformChooser = PluginChooser[ScanPointTransform].createFromList(
            [Scan._createTransformEntry(xform) for xform in ScanPointTransform])

    @classmethod
    def createInstance(cls, rng: numpy.random.Generator, settings: ScanSettings,
                       scanPointSequence: ScanPointSequence) -> TransformedScanPointSequence:
        scan = cls(rng, settings, scanPointSequence)
        settings.transform.addObserver(scan)
        scan._transformChooser.addObserver(scan)
        scan._syncTransformFromSettings()
        return scan

    def __getitem__(self, index: int) -> ScanPoint:
        scanPoint = self._scanPointSequence[index]

        if self._settings.jitterRadiusInMeters.value > 0:
            rad = Decimal(self._rng.uniform())
            dirX = Decimal(self._rng.normal())
            dirY = Decimal(self._rng.normal())

            scalar = self._settings.jitterRadiusInMeters.value \
                    * (rad / (dirX ** 2 + dirY ** 2)).sqrt()
            scanPoint = ScanPoint(scanPoint.x + scalar * dirX, scanPoint.y + scalar * dirY)

        transform = self._transformChooser.getCurrentStrategy()
        return transform(scanPoint)

    def __len__(self) -> int:
        return len(self._scanPointSequence)

    def getTransformList(self) -> list[str]:
        return self._transformChooser.getDisplayNameList()

    def getTransform(self) -> str:
        return self._transformChooser.getCurrentDisplayName()

    def setTransform(self, name: str) -> None:
        self._transformChooser.setFromDisplayName(name)

    def _syncTransformFromSettings(self) -> None:
        self._transformChooser.setFromSimpleName(self._settings.transform.value)

    def _syncTransformToSettings(self) -> None:
        self._updateBoundingBox()
        self._settings.transform.value = self._transformChooser.getCurrentSimpleName()
        self.notifyObservers()

    def update(self, observable: Observable) -> None:
        if observable is self._settings.transform:
            self._syncTransformFromSettings()
        elif observable is self._transformChooser:
            self._syncTransformToSettings()

from __future__ import annotations
from typing import Optional

from ...api.geometry import Box2D, Interval
from ...api.observer import Observable, Observer
from .selected import SelectedScan
from .settings import ScanSettings


class ScanSizer(Observable, Observer):

    def __init__(self, settings: ScanSettings, scan: SelectedScan) -> None:
        super().__init__()
        self._settings = settings
        self._scan = scan

    @classmethod
    def createInstance(cls, settings: ScanSettings, scan: SelectedScan) -> ScanSizer:
        sizer = cls(settings, scan)
        settings.addObserver(sizer)
        scan.addObserver(sizer)
        return sizer

    def getBoundingBoxInMeters(self) -> Optional[Box2D[float]]:
        boundingBoxInMeters: Optional[Box2D[float]] = None

        if self._settings.expandBoundingBox.value:
            rangeXInMeters = Interval[float](
                float(self._settings.boundingBoxMinimumXInMeters.value),
                float(self._settings.boundingBoxMaximumXInMeters.value),
            )
            rangeYInMeters = Interval[float](
                float(self._settings.boundingBoxMinimumYInMeters.value),
                float(self._settings.boundingBoxMaximumYInMeters.value),
            )
            boundingBoxInMeters = Box2D[float](rangeXInMeters, rangeYInMeters)

        selectedScan = self._scan.getSelectedItem()

        if selectedScan is None:
            return boundingBoxInMeters

        it = iter(selectedScan.values())

        try:
            point = next(it)
        except StopIteration:
            return boundingBoxInMeters

        if boundingBoxInMeters is None:
            rangeXInMeters = Interval[float](point.x, point.x)
            rangeYInMeters = Interval[float](point.y, point.y)
            boundingBoxInMeters = Box2D[float](rangeXInMeters, rangeYInMeters)
        else:
            boundingBoxInMeters = boundingBoxInMeters.hull(point.x, point.y)

        for point in it:
            boundingBoxInMeters = boundingBoxInMeters.hull(point.x, point.y)

        # TODO cache this
        return boundingBoxInMeters

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notifyObservers()
        elif observable is self._scan:
            self.notifyObservers()

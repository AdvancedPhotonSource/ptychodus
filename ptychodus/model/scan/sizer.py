from __future__ import annotations
from decimal import Decimal
from typing import Optional

from ...api.geometry import Box2D, Interval
from ...api.observer import Observable, Observer
from ...api.scan import ScanPoint
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

    def getBoundingBoxInMeters(self) -> Optional[Box2D[Decimal]]:
        boundingBoxInMeters: Optional[Box2D[Decimal]] = None

        if self._settings.expandBoundingBox.value:
            rangeXInMeters = Interval[Decimal](
                self._settings.boundingBoxMinimumXInMeters.value,
                self._settings.boundingBoxMaximumXInMeters.value,
            )
            rangeYInMeters = Interval[Decimal](
                self._settings.boundingBoxMinimumYInMeters.value,
                self._settings.boundingBoxMaximumYInMeters.value,
            )
            boundingBoxInMeters = Box2D[Decimal](rangeXInMeters, rangeYInMeters)

        selectedScan = self._scan.getSelectedItem()

        if selectedScan is None:
            return boundingBoxInMeters

        it = iter(selectedScan.values())

        try:
            point = next(it)
        except StopIteration:
            return boundingBoxInMeters

        if boundingBoxInMeters is None:
            rangeXInMeters = Interval[Decimal](point.x, point.x)
            rangeYInMeters = Interval[Decimal](point.y, point.y)
            boundingBoxInMeters = Box2D[Decimal](rangeXInMeters, rangeYInMeters)
        else:
            boundingBoxInMeters = boundingBoxInMeters.hull(point.x, point.y)

        for point in it:
            boundingBoxInMeters = boundingBoxInMeters.hull(point.x, point.y)

        # TODO cache this
        return boundingBoxInMeters

    def getMidpointInMeters(self) -> ScanPoint:
        bbox = self.getBoundingBoxInMeters()
        return ScanPoint(Decimal(), Decimal()) if bbox is None else bbox.midpoint

    def update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notifyObservers()
        elif observable is self._scan:
            self.notifyObservers()

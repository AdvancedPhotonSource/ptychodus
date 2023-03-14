from __future__ import annotations
from decimal import Decimal, ROUND_CEILING

from ...api.geometry import Interval
from ...api.image import ImageExtent
from ...api.observer import Observable, Observer
from ...api.scan import Scan
from ..probe import Apparatus, ProbeSizer


class ObjectSizer(Observable, Observer):

    def __init__(self, apparatus: Apparatus, scan: Scan, probeSizer: ProbeSizer) -> None:
        super().__init__()
        self._apparatus = apparatus
        self._scan = scan
        self._probeSizer = probeSizer

    @classmethod
    def createInstance(cls, apparatus: Apparatus, scan: Scan,
                       probeSizer: ProbeSizer) -> ObjectSizer:
        sizer = cls(apparatus, scan, probeSizer)
        apparatus.addObserver(sizer)
        scan.addObserver(sizer)
        probeSizer.addObserver(sizer)
        return sizer

    def getScanExtent(self) -> ImageExtent:
        scanExtent = ImageExtent(0, 0)
        xint_m = None
        yint_m = None

        for point in self._scan.values():
            if xint_m and yint_m:
                xint_m.hull(point.x)
                yint_m.hull(point.y)
            else:
                xint_m = Interval[Decimal](point.x, point.x)
                yint_m = Interval[Decimal](point.y, point.y)

        if xint_m and yint_m:
            xint_px = xint_m.length / self._apparatus.getObjectPlanePixelSizeXInMeters()
            yint_px = yint_m.length / self._apparatus.getObjectPlanePixelSizeYInMeters()

            scanExtent = ImageExtent(width=int(xint_px.to_integral_exact(rounding=ROUND_CEILING)),
                                     height=int(yint_px.to_integral_exact(rounding=ROUND_CEILING)))

        return scanExtent

    def getPaddingExtent(self) -> ImageExtent:
        return 2 * (self._probeSizer.getProbeExtent() // 2)

    def getObjectExtent(self) -> ImageExtent:
        # FIXME add settings to override minimum object extent
        return self.getScanExtent() + self.getPaddingExtent()

    def update(self, observable: Observable) -> None:
        if observable is self._apparatus:
            self.notifyObservers()
        elif observable is self._scan:
            self.notifyObservers()
        elif observable is self._probeSizer:
            self.notifyObservers()

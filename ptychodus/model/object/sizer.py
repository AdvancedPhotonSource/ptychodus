from __future__ import annotations
from decimal import Decimal, ROUND_CEILING

from ...api.geometry import Interval
from ...api.image import ImageExtent
from ...api.observer import Observable, Observer
from ..data import CropSizer, Detector
from ..probe import ProbeSizer
from ..scan import Scan


class ObjectSizer(Observable, Observer):

    def __init__(self, detector: Detector, cropSizer: CropSizer, scan: Scan,
                 probeSizer: ProbeSizer) -> None:
        super().__init__()
        self._detector = detector
        self._cropSizer = cropSizer
        self._scan = scan
        self._probeSizer = probeSizer

    @classmethod
    def createInstance(cls, detector: Detector, cropSizer: CropSizer, scan: Scan,
                       probeSizer: ProbeSizer) -> ObjectSizer:
        sizer = cls(detector, cropSizer, scan, probeSizer)
        detector.addObserver(sizer)
        cropSizer.addObserver(sizer)
        scan.addObserver(sizer)
        probeSizer.addObserver(sizer)
        return sizer

    @property
    def _lambdaZ_m2(self) -> Decimal:
        return self._probeSizer.getWavelengthInMeters() \
                * self._detector.getDetectorDistanceInMeters()

    def getPixelSizeXInMeters(self) -> Decimal:
        extentXInMeters = self._cropSizer.getExtentXInPixels() \
                * self._detector.getPixelSizeXInMeters()
        return self._lambdaZ_m2 / extentXInMeters

    def getPixelSizeYInMeters(self) -> Decimal:
        extentYInMeters = self._cropSizer.getExtentYInPixels() \
                * self._detector.getPixelSizeYInMeters()
        return self._lambdaZ_m2 / extentYInMeters

    def getScanExtent(self) -> ImageExtent:
        scanExtent = ImageExtent(0, 0)
        xint_m = None
        yint_m = None

        for point in self._scan:
            if xint_m and yint_m:
                xint_m.hull(point.x)
                yint_m.hull(point.y)
            else:
                xint_m = Interval[Decimal](point.x, point.x)
                yint_m = Interval[Decimal](point.y, point.y)

        if xint_m and yint_m:
            xint_px = xint_m.length / self.getPixelSizeXInMeters()
            yint_px = yint_m.length / self.getPixelSizeYInMeters()

            scanExtent = ImageExtent(width=int(xint_px.to_integral_exact(rounding=ROUND_CEILING)),
                                     height=int(yint_px.to_integral_exact(rounding=ROUND_CEILING)))

        return scanExtent

    def getPaddingExtent(self) -> ImageExtent:
        return 2 * (self._probeSizer.getProbeExtent() // 2)

    def getObjectExtent(self) -> ImageExtent:
        return self.getScanExtent() + self.getPaddingExtent()

    def update(self, observable: Observable) -> None:
        if observable is self._detector:
            self.notifyObservers()
        elif observable is self._cropSizer:
            self.notifyObservers()
        elif observable is self._scan:
            self.notifyObservers()
        elif observable is self._probeSizer:
            self.notifyObservers()

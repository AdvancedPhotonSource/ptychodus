import numpy

from ...api.geometry import Box2D, Interval
from ...api.observer import Observable
from ...api.parametric import ParameterRepository
from ...api.patterns import ImageExtent, PixelGeometry
from ..metadata import MetadataRepositoryItem
from ..patterns import PatternSizer
from ..scan import ScanRepositoryItem


class ExperimentSizer(ParameterRepository):

    def __init__(self, metadata: MetadataRepositoryItem, scan: ScanRepositoryItem,
                 patternSizer: PatternSizer) -> None:
        super().__init__('Sizer')
        self._metadata = metadata
        self._scan = scan
        self._patternSizer = patternSizer

        self._metadata.addObserver(self)
        self._scan.addObserver(self)
        self._patternSizer.addObserver(self)

        # FIXME to GUI
        self.expandScanBoundingBox = self._registerBooleanParameter('ExpandScanBoundingBox', False)
        self.scanBoundingBoxMinimumXInMeters = self._registerRealParameter(
            'ScanBoundingBoxMinimumXInMeters', 0.)
        self.scanBoundingBoxMaximumXInMeters = self._registerRealParameter(
            'ScanBoundingBoxMaximumXInMeters', 1.e-5)
        self.scanBoundingBoxMinimumYInMeters = self._registerRealParameter(
            'ScanBoundingBoxMinimumYInMeters', 0.)
        self.scanBoundingBoxMaximumYInMeters = self._registerRealParameter(
            'ScanBoundingBoxMaximumYInMeters', 1.e-5)

    @property
    def _lambdaZInSquareMeters(self) -> float:
        lambdaInMeters = self._metadata.getProbeWavelengthInMeters()
        zInMeters = self._metadata.detectorObjectDistanceInMeters.getValue()
        return lambdaInMeters * zInMeters

    def getObjectPlanePixelGeometry(self) -> PixelGeometry:
        lambdaZInSquareMeters = self._lambdaZInSquareMeters
        return PixelGeometry(
            widthInMeters=lambdaZInSquareMeters / self._patternSizer.getWidthInMeters(),
            heightInMeters=lambdaZInSquareMeters / self._patternSizer.getHeightInMeters(),
        )

    def getFresnelNumber(self) -> float:
        widthInMeters = self._patternSizer.getWidthInMeters()
        heightInMeters = self._patternSizer.getHeightInMeters()
        sizeInMeters = max(widthInMeters, heightInMeters)
        return sizeInMeters**2 / self._lambdaZInSquareMeters

    def getScanExtent(self) -> ImageExtent:
        bbox = self._scan.getBoundingBoxInMeters()
        widthInPixels = 0
        heightInPixels = 0

        if self.expandScanBoundingBox.getValue():
            rangeX = Interval[float](
                self.scanBoundingBoxMinimumXInMeters.getValue(),
                self.scanBoundingBoxMaximumXInMeters.getValue(),
            )
            rangeY = Interval[float](
                self.scanBoundingBoxMinimumYInMeters.getValue(),
                self.scanBoundingBoxMaximumYInMeters.getValue(),
            )

            if bbox is None:
                bbox = Box2D(rangeX, rangeY)
            else:
                bbox = bbox.hull(rangeX, rangeY)

        if bbox is not None:
            pixelWidthInMeters = self._patternSizer.getPixelWidthInMeters()
            pixelHeightInMeters = self._patternSizer.getPixelHeightInMeters()

            widthInPixels = int(numpy.ceil(bbox.rangeX.length / pixelWidthInMeters))
            heightInPixels = int(numpy.ceil(bbox.rangeY.length / pixelHeightInMeters))

        return ImageExtent(widthInPixels, heightInPixels)

    def getProbeExtent(self) -> ImageExtent:
        return self._patternSizer.getImageExtent()

    def isProbeExtentValid(self, extent: ImageExtent) -> bool:
        expectedExtent = self.getProbeExtent()
        return (extent == expectedExtent)

    def getObjectExtent(self) -> ImageExtent:
        scanExtent = self.getScanExtent()
        probeExtent = self.getProbeExtent()

        return ImageExtent(
            widthInPixels=scanExtent.widthInPixels + probeExtent.widthInPixels,
            heightInPixels=scanExtent.heightInPixels + probeExtent.heightInPixels,
        )

    def isObjectExtentValid(self, extent: ImageExtent) -> bool:
        expectedExtent = self.getObjectExtent()
        widthIsBigEnough = (extent.widthInPixels >= expectedExtent.widthInPixels)
        heightIsBigEnough = (extent.heightInPixels >= expectedExtent.heightInPixels)
        return (widthIsBigEnough and heightIsBigEnough)

    def update(self, observable: Observable) -> None:
        if observable is self._metadata:
            self.notifyObservers()
        elif observable is self._scan:
            self.notifyObservers()
        elif observable is self._patternSizer:
            self.notifyObservers()

import numpy

from ...api.observer import Observable, Observer
from ...api.patterns import ImageExtent, PixelGeometry
from ..metadata import MetadataRepositoryItem
from ..patterns import PatternSizer
from ..scan import ScanRepositoryItem


class ExperimentSizer(Observable, Observer):

    def __init__(self, metadata: MetadataRepositoryItem, scan: ScanRepositoryItem,
                 patternSizer: PatternSizer) -> None:
        super().__init__()
        # FIXME pull defaults from settings; how to use diffraction pattern metadata?
        # FIXME getters/setters for expand bounding box
        self._metadata = metadata
        self._scan = scan
        self._patternSizer = patternSizer

        self._metadata.addObserver(self)
        self._scan.addObserver(self)
        self._patternSizer.addObserver(self)

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
        # TODO support non-square pixels
        widthInMeters = self._patternSizer.getWidthInMeters()
        return widthInMeters**2 / self._lambdaZInSquareMeters

    def getScanImageExtent(self) -> ImageExtent:
        bbox = self._scan.getBoundingBoxInMeters()
        widthInPixels = 0
        heightInPixels = 0

        if bbox is not None:
            pixelWidthInMeters = self._patternSizer.getPixelWidthInMeters()
            widthInPixels = int(numpy.ceil(bbox.rangeX.width / float(pixelWidthInMeters)))
            pixelHeightInMeters = self._patternSizer.getPixelHeightInMeters()
            heightInPixels = int(numpy.ceil(bbox.rangeY.width / float(pixelHeightInMeters)))

        return ImageExtent(widthInPixels, heightInPixels)

    def getProbeImageExtent(self) -> ImageExtent:
        return self._patternSizer.getImageExtent()

    def getObjectImageExtent(self) -> ImageExtent:
        scanExtent = self.getScanImageExtent()
        probeExtent = self.getProbeImageExtent()

        return ImageExtent(
            widthInPixels=scanExtent.widthInPixels + probeExtent.widthInPixels,
            heightInPixels=scanExtent.heightInPixels + probeExtent.heightInPixels,
        )

    def update(self, observable: Observable) -> None:
        if observable is self._metadata:
            self.notifyObservers()
        elif observable is self._scan:
            self.notifyObservers()
        elif observable is self._patternSizer:
            self.notifyObservers()

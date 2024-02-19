import numpy

from ...api.object import ObjectGeometry, ObjectGeometryProvider
from ...api.observer import Observable
from ...api.parametric import ParameterRepository
from ...api.probe import ProbeGeometry, ProbeGeometryProvider
from ...api.scan import ScanBoundingBox
from ..metadata import MetadataRepositoryItem
from ..patterns import PatternSizer
from ..scan import ScanRepositoryItem


class ProductGeometry(ParameterRepository, ProbeGeometryProvider, ObjectGeometryProvider):

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
    def probeWavelengthInMeters(self) -> float:
        # Source: https://physics.nist.gov/cuu/Constants/index.html
        planckConstant_eV_per_Hz = 4.135667696e-15
        lightSpeedInMetersPerSecond = 299792458
        hc_eVm = planckConstant_eV_per_Hz * lightSpeedInMetersPerSecond
        return hc_eVm / self._metadata.probeEnergyInElectronVolts.getValue()

    @property
    def _lambdaZInSquareMeters(self) -> float:
        return self.probeWavelengthInMeters * self._metadata.detectorDistanceInMeters.getValue()

    @property
    def objectPlanePixelWidthInMeters(self) -> float:
        return self._lambdaZInSquareMeters / self._patternSizer.getWidthInMeters()

    @property
    def objectPlanePixelHeightInMeters(self) -> float:
        return self._lambdaZInSquareMeters / self._patternSizer.getHeightInMeters()

    @property
    def fresnelNumber(self) -> float:
        widthInMeters = self._patternSizer.getWidthInMeters()
        heightInMeters = self._patternSizer.getHeightInMeters()
        sizeInMeters = max(widthInMeters, heightInMeters)
        return sizeInMeters**2 / self._lambdaZInSquareMeters

    def getScanBoundingBox(self) -> ScanBoundingBox | None:
        bbox = self._scan.getBoundingBox()

        if self.expandScanBoundingBox.getValue():
            expandedBBox = ScanBoundingBox(
                minimumXInMeters=self.scanBoundingBoxMinimumXInMeters.getValue(),
                maximumXInMeters=self.scanBoundingBoxMaximumXInMeters.getValue(),
                minimumYInMeters=self.scanBoundingBoxMinimumYInMeters.getValue(),
                maximumYInMeters=self.scanBoundingBoxMaximumYInMeters.getValue(),
            )
            bbox = expandedBBox if bbox is None else bbox.hull(expandedBBox)

        return bbox

    def getProbeGeometry(self) -> ProbeGeometry:
        extent = self._patternSizer.getImageExtent()
        return ProbeGeometry(
            widthInPixels=extent.widthInPixels,
            heightInPixels=extent.heightInPixels,
            pixelWidthInMeters=self.objectPlanePixelWidthInMeters,
            pixelHeightInMeters=self.objectPlanePixelHeightInMeters,
        )

    def isProbeGeometryValid(self, geometry: ProbeGeometry) -> bool:
        expected = self.getProbeGeometry()
        widthIsValid = (geometry.pixelWidthInMeters > 0.
                        and geometry.widthInMeters == expected.widthInMeters)
        heightIsValid = (geometry.pixelHeightInMeters > 0.
                         and geometry.heightInMeters == expected.heightInMeters)
        return (widthIsValid and heightIsValid)

    def getObjectGeometry(self) -> ObjectGeometry:
        probeGeometry = self.getProbeGeometry()
        widthInMeters = probeGeometry.widthInMeters
        heightInMeters = probeGeometry.heightInMeters
        centerXInMeters = 0.
        centerYInMeters = 0.

        scanBoundingBox = self.getScanBoundingBox()

        if scanBoundingBox is not None:
            widthInMeters += scanBoundingBox.widthInMeters
            heightInMeters += scanBoundingBox.heightInMeters
            centerXInMeters = scanBoundingBox.centerXInMeters
            centerYInMeters = scanBoundingBox.centerYInMeters

        widthInPixels = widthInMeters / self.objectPlanePixelWidthInMeters
        heightInPixels = heightInMeters / self.objectPlanePixelHeightInMeters

        return ObjectGeometry(
            widthInPixels=int(numpy.ceil(widthInPixels)),
            heightInPixels=int(numpy.ceil(heightInPixels)),
            pixelWidthInMeters=self.objectPlanePixelWidthInMeters,
            pixelHeightInMeters=self.objectPlanePixelHeightInMeters,
            centerXInMeters=centerXInMeters,
            centerYInMeters=centerYInMeters,
        )

    def isObjectGeometryValid(self, geometry: ObjectGeometry) -> bool:
        expectedGeometry = self.getObjectGeometry()
        pixelSizeIsValid = (geometry.pixelWidthInMeters > 0. and geometry.pixelHeightInMeters > 0.)
        return pixelSizeIsValid and geometry.contains(expectedGeometry)

    def update(self, observable: Observable) -> None:
        if observable is self._metadata:
            self.notifyObservers()
        elif observable is self._scan:
            self.notifyObservers()
        elif observable is self._patternSizer:
            self.notifyObservers()
        else:
            super().update(observable)

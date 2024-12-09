import numpy

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.object import ObjectGeometry, ObjectGeometryProvider
from ptychodus.api.observer import Observable, Observer
from ptychodus.api.probe import ProbeGeometry, ProbeGeometryProvider
from ptychodus.api.product import (
    ELECTRON_VOLT_J,
    LIGHT_SPEED_M_PER_S,
    PLANCK_CONSTANT_J_PER_HZ,
)

from ..patterns import PatternSizer
from .metadata import MetadataRepositoryItem
from .scan import ScanRepositoryItem


class ProductGeometry(ProbeGeometryProvider, ObjectGeometryProvider, Observable, Observer):
    def __init__(
        self,
        patternSizer: PatternSizer,
        metadata: MetadataRepositoryItem,
        scan: ScanRepositoryItem,
    ) -> None:
        super().__init__()
        self._patternSizer = patternSizer
        self._metadata = metadata
        self._scan = scan

        self._patternSizer.addObserver(self)
        self._metadata.addObserver(self)
        self._scan.addObserver(self)

    @property
    def probePhotonCount(self) -> float:
        return self._metadata.probePhotonCount.getValue()

    @property
    def probeEnergyInJoules(self) -> float:
        return self._metadata.probeEnergyInElectronVolts.getValue() * ELECTRON_VOLT_J

    @property
    def probeWavelengthInMeters(self) -> float:
        hc_Jm = PLANCK_CONSTANT_J_PER_HZ * LIGHT_SPEED_M_PER_S

        try:
            return hc_Jm / self.probeEnergyInJoules
        except ZeroDivisionError:
            return 0.0

    @property
    def probeWavelengthsPerMeter(self) -> float:
        """wavenumber"""
        return 1.0 / self.probeWavelengthInMeters

    @property
    def probeRadiansPerMeter(self) -> float:
        """angular wavenumber"""
        return 2.0 * numpy.pi / self.probeWavelengthInMeters

    @property
    def probePhotonsPerSecond(self) -> float:
        try:
            return self.probePhotonCount / self._metadata.exposureTimeInSeconds.getValue()
        except ZeroDivisionError:
            return 0.0

    @property
    def probePowerInWatts(self) -> float:
        return self.probeEnergyInJoules * self.probePhotonsPerSecond

    @property
    def detectorDistanceInMeters(self) -> float:
        return self._metadata.detectorDistanceInMeters.getValue()

    @property
    def _lambdaZInSquareMeters(self) -> float:
        return self.probeWavelengthInMeters * self.detectorDistanceInMeters

    @property
    def objectPlanePixelWidthInMeters(self) -> float:
        return self._lambdaZInSquareMeters / self._patternSizer.getWidthInMeters()

    @property
    def objectPlanePixelHeightInMeters(self) -> float:
        return self._lambdaZInSquareMeters / self._patternSizer.getHeightInMeters()

    def getPixelGeometry(self) -> PixelGeometry:
        return PixelGeometry(
            widthInMeters=self.objectPlanePixelWidthInMeters,
            heightInMeters=self.objectPlanePixelHeightInMeters,
        )

    @property
    def fresnelNumber(self) -> float:
        widthInMeters = self._patternSizer.getWidthInMeters()
        heightInMeters = self._patternSizer.getHeightInMeters()
        areaInSquareMeters = widthInMeters * heightInMeters
        return areaInSquareMeters / self._lambdaZInSquareMeters

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
        widthIsValid = (
            geometry.pixelWidthInMeters > 0.0 and geometry.widthInMeters == expected.widthInMeters
        )
        heightIsValid = (
            geometry.pixelHeightInMeters > 0.0
            and geometry.heightInMeters == expected.heightInMeters
        )
        return widthIsValid and heightIsValid

    def getObjectGeometry(self) -> ObjectGeometry:
        probeGeometry = self.getProbeGeometry()
        widthInMeters = probeGeometry.widthInMeters
        heightInMeters = probeGeometry.heightInMeters
        centerXInMeters = 0.0
        centerYInMeters = 0.0

        scanBoundingBox = self._scan.getBoundingBox()

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
        pixelSizeIsValid = geometry.pixelWidthInMeters > 0.0 and geometry.pixelHeightInMeters > 0.0
        return pixelSizeIsValid and geometry.contains(expectedGeometry)

    def update(self, observable: Observable) -> None:
        if observable is self._metadata:
            self.notifyObservers()
        elif observable is self._scan:
            self.notifyObservers()
        elif observable is self._patternSizer:
            self.notifyObservers()

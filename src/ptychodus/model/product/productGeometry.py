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

        self._patternSizer.add_observer(self)
        self._metadata.add_observer(self)
        self._scan.add_observer(self)

    @property
    def probe_photon_count(self) -> float:
        return self._metadata.probePhotonCount.get_value()

    @property
    def probeEnergyInJoules(self) -> float:
        return self._metadata.probeEnergyInElectronVolts.get_value() * ELECTRON_VOLT_J

    @property
    def probe_wavelength_m(self) -> float:
        hc_Jm = PLANCK_CONSTANT_J_PER_HZ * LIGHT_SPEED_M_PER_S

        try:
            return hc_Jm / self.probeEnergyInJoules
        except ZeroDivisionError:
            return 0.0

    @property
    def probeWavelengthsPerMeter(self) -> float:
        """wavenumber"""
        return 1.0 / self.probe_wavelength_m

    @property
    def probeRadiansPerMeter(self) -> float:
        """angular wavenumber"""
        return 2.0 * numpy.pi / self.probe_wavelength_m

    @property
    def probePhotonsPerSecond(self) -> float:
        try:
            return self.probe_photon_count / self._metadata.exposureTimeInSeconds.get_value()
        except ZeroDivisionError:
            return 0.0

    @property
    def probe_power_W(self) -> float:
        return self.probeEnergyInJoules * self.probePhotonsPerSecond

    @property
    def detector_distance_m(self) -> float:
        return self._metadata.detectorDistanceInMeters.get_value()

    @property
    def _lambdaZInSquareMeters(self) -> float:
        return self.probe_wavelength_m * self.detector_distance_m

    @property
    def objectPlanePixelWidthInMeters(self) -> float:
        return self._lambdaZInSquareMeters / self._patternSizer.get_processed_width_m()

    @property
    def objectPlanePixelHeightInMeters(self) -> float:
        return self._lambdaZInSquareMeters / self._patternSizer.get_processed_height_m()

    def get_detector_pixel_geometry(self):
        return self._patternSizer.get_processed_pixel_geometry()

    def get_object_plane_pixel_geometry(self) -> PixelGeometry:
        return PixelGeometry(
            width_m=self.objectPlanePixelWidthInMeters,
            height_m=self.objectPlanePixelHeightInMeters,
        )

    @property
    def fresnelNumber(self) -> float:
        widthInMeters = self._patternSizer.get_processed_width_m()
        heightInMeters = self._patternSizer.get_processed_height_m()
        areaInSquareMeters = widthInMeters * heightInMeters
        return areaInSquareMeters / self._lambdaZInSquareMeters

    def get_probe_geometry(self) -> ProbeGeometry:
        extent = self._patternSizer.get_processed_image_extent()
        return ProbeGeometry(
            width_px=extent.width_px,
            height_px=extent.height_px,
            pixel_width_m=self.objectPlanePixelWidthInMeters,
            pixel_height_m=self.objectPlanePixelHeightInMeters,
        )

    def isProbeGeometryValid(self, geometry: ProbeGeometry) -> bool:
        expected = self.get_probe_geometry()
        widthIsValid = geometry.pixel_width_m > 0.0 and geometry.width_m == expected.width_m
        heightIsValid = geometry.pixel_height_m > 0.0 and geometry.height_m == expected.height_m
        return widthIsValid and heightIsValid

    def get_object_geometry(self) -> ObjectGeometry:
        probeGeometry = self.get_probe_geometry()
        widthInMeters = probeGeometry.width_m
        heightInMeters = probeGeometry.height_m
        centerXInMeters = 0.0
        centerYInMeters = 0.0

        scanBoundingBox = self._scan.getBoundingBox()

        if scanBoundingBox is not None:
            widthInMeters += scanBoundingBox.width_m
            heightInMeters += scanBoundingBox.height_m
            centerXInMeters = scanBoundingBox.center_x_m
            centerYInMeters = scanBoundingBox.center_y_m

        widthInPixels = widthInMeters / self.objectPlanePixelWidthInMeters
        heightInPixels = heightInMeters / self.objectPlanePixelHeightInMeters

        return ObjectGeometry(
            width_px=int(numpy.ceil(widthInPixels)),
            height_px=int(numpy.ceil(heightInPixels)),
            pixel_width_m=self.objectPlanePixelWidthInMeters,
            pixel_height_m=self.objectPlanePixelHeightInMeters,
            center_x_m=centerXInMeters,
            center_y_m=centerYInMeters,
        )

    def isObjectGeometryValid(self, geometry: ObjectGeometry) -> bool:
        expectedGeometry = self.get_object_geometry()
        pixelSizeIsValid = geometry.pixel_width_m > 0.0 and geometry.pixel_height_m > 0.0
        return pixelSizeIsValid and geometry.contains(expectedGeometry)

    def _update(self, observable: Observable) -> None:
        if observable is self._metadata:
            self.notify_observers()
        elif observable is self._scan:
            self.notify_observers()
        elif observable is self._patternSizer:
            self.notify_observers()

from __future__ import annotations

import numpy

from ptychodus.api.probe import Probe, ProbeGeometryProvider
from ptychodus.api.propagator import FresnelTransformPropagator, PropagatorParameters

from ...patterns import ActiveDiffractionDataset, Detector
from .builder import ProbeBuilder


class AveragePatternProbeBuilder(ProbeBuilder):
    def __init__(self, detector: Detector, patterns: ActiveDiffractionDataset) -> None:
        super().__init__('average_pattern')
        self._detector = detector
        self._patterns = patterns

    def copy(self) -> AveragePatternProbeBuilder:
        return AveragePatternProbeBuilder(self._detector, self._patterns)

    def build(self, geometryProvider: ProbeGeometryProvider) -> Probe:
        geometry = geometryProvider.getProbeGeometry()
        detectorIntensity = numpy.average(self._patterns.getAssembledData(), axis=0)

        pixelGeometry = self._detector.getPixelGeometry()
        propagatorParameters = PropagatorParameters(
            wavelength_m=geometryProvider.probeWavelengthInMeters,
            width_px=detectorIntensity.shape[-1],
            height_px=detectorIntensity.shape[-2],
            pixel_width_m=pixelGeometry.widthInMeters,
            pixel_height_m=pixelGeometry.heightInMeters,
            propagation_distance_m=-geometryProvider.detectorDistanceInMeters,
        )
        propagator = FresnelTransformPropagator(propagatorParameters)
        array = propagator.propagate(numpy.sqrt(detectorIntensity).astype(complex))

        return Probe(
            array=self.normalize(array),
            pixelWidthInMeters=geometry.pixelWidthInMeters,
            pixelHeightInMeters=geometry.pixelHeightInMeters,
        )

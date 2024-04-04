from __future__ import annotations

import numpy

from ptychodus.api.probe import Probe, ProbeGeometryProvider

from ...patterns import ActiveDiffractionDataset, Detector
from ...propagator import fresnel_propagator
from .builder import ProbeBuilder


class AveragePatternProbeBuilder(ProbeBuilder):

    def __init__(self, detector: Detector, patterns: ActiveDiffractionDataset) -> None:
        super().__init__('average_pattern')
        self._detector = detector
        self._patterns = patterns

    def copy(self) -> AveragePatternProbeBuilder:
        return AveragePatternProbeBuilder(self._detector, self._patterns)

    def build(self, geometryProvider: ProbeGeometryProvider) -> Probe:
        detectorPixelSizeInMeters = float(self._detector.getPixelWidthInMeters())
        geometry = geometryProvider.getProbeGeometry()
        detectorIntensity = numpy.average(self._patterns.getAssembledData(), axis=0)
        probeIntensity = fresnel_propagator(detectorIntensity.astype(complex),
                                            detectorPixelSizeInMeters,
                                            -geometryProvider.detectorDistanceInMeters,
                                            geometryProvider.probeWavelengthInMeters)
        array = numpy.sqrt(probeIntensity)

        return Probe(
            array=self.normalize(array),
            pixelWidthInMeters=geometry.pixelWidthInMeters,
            pixelHeightInMeters=geometry.pixelHeightInMeters,
        )

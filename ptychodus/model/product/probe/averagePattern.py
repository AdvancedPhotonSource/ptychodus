from __future__ import annotations

import numpy

from ptychodus.api.probe import Probe, ProbeGeometryProvider

from ...patterns import ActiveDiffractionDataset
from .builder import ProbeBuilder


class AveragePatternProbeBuilder(ProbeBuilder):

    def __init__(self, patterns: ActiveDiffractionDataset) -> None:
        super().__init__('average_pattern')
        self._patterns = patterns

    def copy(self) -> AveragePatternProbeBuilder:
        return AveragePatternProbeBuilder(self._patterns)

    def build(self, geometryProvider: ProbeGeometryProvider) -> Probe:
        geometry = geometryProvider.getProbeGeometry()
        intensity = numpy.average(self._patterns.getAssembledData(), axis=0)
        # FIXME ifft to backpropagate to object plane
        array = numpy.sqrt(intensity, dtype=complex)

        return Probe(
            array=self.normalize(array),
            pixelWidthInMeters=geometry.pixelWidthInMeters,
            pixelHeightInMeters=geometry.pixelHeightInMeters,
        )

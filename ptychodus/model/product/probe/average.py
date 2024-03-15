from __future__ import annotations

import numpy

from ptychodus.api.probe import Probe, ProbeGeometryProvider

from ...patterns import ActiveDiffractionDataset
from .builder import ProbeBuilder


class AverageProbeBuilder(ProbeBuilder):

    def __init__(self, patterns: ActiveDiffractionDataset,
                 geometryProvider: ProbeGeometryProvider) -> None:
        super().__init__('Average')
        self._patterns = patterns
        self._geometryProvider = geometryProvider

    def copy(self, geometryProvider: ProbeGeometryProvider) -> AverageProbeBuilder:
        return AverageProbeBuilder(self._patterns, geometryProvider)

    def build(self) -> Probe:
        geometry = self._geometryProvider.getProbeGeometry()
        intensity = numpy.average(self._patterns.getAssembledData(), axis=0)
        # FIXME ifft to backpropagate to object plane
        array = numpy.sqrt(intensity, dtype=complex)

        return Probe(
            array=self.normalize(array),
            pixelWidthInMeters=geometry.pixelWidthInMeters,
            pixelHeightInMeters=geometry.pixelHeightInMeters,
        )

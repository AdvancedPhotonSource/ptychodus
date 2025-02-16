from __future__ import annotations

import numpy

from ptychodus.api.probe import Probe, ProbeGeometryProvider
from ptychodus.api.propagator import FresnelTransformPropagator, PropagatorParameters

from ...patterns import AssembledDiffractionDataset, PatternSizer
from .builder import ProbeBuilder
from .settings import ProbeSettings


class AveragePatternProbeBuilder(ProbeBuilder):
    def __init__(
        self,
        settings: ProbeSettings,
        patternSizer: PatternSizer,
        dataset: AssembledDiffractionDataset,
    ) -> None:
        super().__init__(settings, 'average_pattern')
        self._settings = settings
        self._patternSizer = patternSizer
        self._dataset = dataset

    def copy(self) -> AveragePatternProbeBuilder:
        return AveragePatternProbeBuilder(self._settings, self._patternSizer, self._dataset)

    def build(self, geometryProvider: ProbeGeometryProvider) -> Probe:
        geometry = geometryProvider.getProbeGeometry()
        detectorIntensity = numpy.average(self._dataset.get_assembled_patterns(), axis=0)

        pixelGeometry = self._patternSizer.getDetectorPixelGeometry()
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
            pixelGeometry=geometry.getPixelGeometry(),
        )

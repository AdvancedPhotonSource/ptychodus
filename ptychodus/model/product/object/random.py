from __future__ import annotations
from collections.abc import Sequence

import numpy

from ptychodus.api.object import Object, ObjectGeometryProvider

from .builder import ObjectBuilder
from .settings import ObjectSettings


class RandomObjectBuilder(ObjectBuilder):

    def __init__(self, rng: numpy.random.Generator, settings: ObjectSettings) -> None:
        super().__init__('random')
        self._rng = rng
        self._settings = settings

        self.extraPaddingX = self._registerIntegerParameter('extra_padding_x',
                                                            settings.extraPaddingX.value,
                                                            minimum=0)
        self.extraPaddingY = self._registerIntegerParameter('extra_padding_y',
                                                            settings.extraPaddingY.value,
                                                            minimum=0)

        self.amplitudeMean = self._registerRealParameter(
            'amplitude_mean',
            float(settings.amplitudeMean.value),
            minimum=0.,
            maximum=1.,
        )
        self.amplitudeDeviation = self._registerRealParameter(
            'amplitude_deviation',
            float(settings.amplitudeDeviation.value),
            minimum=0.,
            maximum=1.,
        )
        self.phaseDeviation = self._registerRealParameter(
            'phase_deviation',
            float(settings.phaseDeviation.value),
            minimum=0.,
            maximum=numpy.pi,
        )

    def copy(self) -> RandomObjectBuilder:
        builder = RandomObjectBuilder(self._rng, self._settings)
        builder.extraPaddingX.setValue(self.extraPaddingX.getValue())
        builder.extraPaddingY.setValue(self.extraPaddingY.getValue())
        builder.amplitudeMean.setValue(self.amplitudeMean.getValue())
        builder.amplitudeDeviation.setValue(self.amplitudeDeviation.getValue())
        builder.phaseDeviation.setValue(self.phaseDeviation.getValue())
        return builder

    def build(self, geometryProvider: ObjectGeometryProvider,
              layerDistanceInMeters: Sequence[float]) -> Object:
        geometry = geometryProvider.getObjectGeometry()
        heightInPixels = geometry.heightInPixels + 2 * self.extraPaddingY.getValue()
        widthInPixels = geometry.widthInPixels + 2 * self.extraPaddingX.getValue()
        objectShape = (len(layerDistanceInMeters), heightInPixels, widthInPixels)

        amplitude = self._rng.normal(
            self.amplitudeMean.getValue(),
            self.amplitudeDeviation.getValue(),
            objectShape,
        )
        phase = self._rng.normal(
            0.,
            self.phaseDeviation.getValue(),
            objectShape,
        )

        return Object(
            array=numpy.clip(amplitude, 0., 1.) * numpy.exp(1j * phase),
            layerDistanceInMeters=layerDistanceInMeters,
            pixelWidthInMeters=geometry.pixelWidthInMeters,
            pixelHeightInMeters=geometry.pixelHeightInMeters,
            centerXInMeters=geometry.centerXInMeters,
            centerYInMeters=geometry.centerYInMeters,
        )

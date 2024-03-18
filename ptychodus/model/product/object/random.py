from __future__ import annotations

import numpy

from ptychodus.api.object import Object, ObjectGeometryProvider

from .builder import ObjectBuilder


class RandomObjectBuilder(ObjectBuilder):

    def __init__(self, rng: numpy.random.Generator,
                 geometryProvider: ObjectGeometryProvider) -> None:
        super().__init__('random')
        self._rng = rng
        self._geometryProvider = geometryProvider

        self.extraPaddingX = self._registerIntegerParameter('extra_padding_x', 1, minimum=0)
        self.extraPaddingY = self._registerIntegerParameter('extra_padding_y', 1, minimum=0)

        self.amplitudeMean = self._registerRealParameter(
            'amplitude_mean',
            0.5,
            minimum=0.,
            maximum=1.,
        )
        self.amplitudeDeviation = self._registerRealParameter(
            'amplitude_deviation',
            0.,
            minimum=0.,
            maximum=1.,
        )
        self.phaseDeviation = self._registerRealParameter(
            'phase_deviation',
            0.,
            minimum=0.,
            maximum=numpy.pi,
        )
        self.numberOfLayers = self._registerIntegerParameter(
            'number_of_layers',
            1,
            minimum=1,
        )
        self.layerDistanceInMeters = self._registerRealParameter(
            'layer_distance_m',
            1.e-6,
            minimum=0.,
        )

    def copy(self, geometryProvider: ObjectGeometryProvider) -> RandomObjectBuilder:
        builder = RandomObjectBuilder(self._rng, geometryProvider)
        builder.extraPaddingX.setValue(self.extraPaddingX.getValue())
        builder.extraPaddingY.setValue(self.extraPaddingY.getValue())
        builder.amplitudeMean.setValue(self.amplitudeMean.getValue())
        builder.amplitudeDeviation.setValue(self.amplitudeDeviation.getValue())
        builder.phaseDeviation.setValue(self.phaseDeviation.getValue())
        builder.numberOfLayers.setValue(self.numberOfLayers.getValue())
        builder.layerDistanceInMeters.setValue(self.layerDistanceInMeters.getValue())
        return builder

    def build(self) -> Object:
        geometry = self._geometryProvider.getObjectGeometry()
        widthInPixels = geometry.widthInPixels + 2 * self.extraPaddingX.getValue()
        heightInPixels = geometry.heightInPixels + 2 * self.extraPaddingY.getValue()
        objectShape = (self.numberOfLayers.getValue(), heightInPixels, widthInPixels)

        amplitude = self._rng.normal(
            self.amplitudeMean.getValue(),
            self.amplitudeDeviation.getValue(),
            objectShape,
        )
        phase = self._rng.normal(
            -self.phaseDeviation.getValue(),
            +self.phaseDeviation.getValue(),
            objectShape,
        )

        # FIXME end with inf
        layerDistances = [self.layerDistanceInMeters.getValue()] * self.numberOfLayers.getValue()

        return Object(
            array=numpy.clip(amplitude, 0., 1.) * numpy.exp(1j * phase),
            layerDistanceInMeters=layerDistances,
            pixelWidthInMeters=geometry.pixelWidthInMeters,
            pixelHeightInMeters=geometry.pixelHeightInMeters,
            centerXInMeters=geometry.centerXInMeters,
            centerYInMeters=geometry.centerYInMeters,
        )

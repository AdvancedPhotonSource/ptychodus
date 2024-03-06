import numpy

from ptychodus.api.object import Object, ObjectGeometryProvider

from .builder import ObjectBuilder


class RandomObjectBuilder(ObjectBuilder):

    def __init__(self, rng: numpy.random.Generator,
                 geometryProvider: ObjectGeometryProvider) -> None:
        super().__init__('Random')
        self._rng = rng
        self._geometryProvider = geometryProvider

        self.extraPaddingX = self._registerIntegerParameter('ExtraPaddingX', 1, minimum=0)
        self.extraPaddingY = self._registerIntegerParameter('ExtraPaddingY', 1, minimum=0)

        self.amplitudeMean = self._registerRealParameter(
            'AmplitudeMean',
            0.5,
            minimum=0.,
            maximum=1.,
        )
        self.amplitudeDeviation = self._registerRealParameter(
            'AmplitudeDeviation',
            0.,
            minimum=0.,
            maximum=1.,
        )
        self.phaseDeviation = self._registerRealParameter(
            'PhaseDeviation',
            0.,
            minimum=0.,
            maximum=numpy.pi,
        )
        self.numberOfLayers = self._registerIntegerParameter(
            'NumberOfLayers',
            1,
            minimum=1,
        )
        self.layerDistanceInMeters = self._registerRealParameter(
            'LayerDistanceInMeters',
            1.e-6,
            minimum=0.,
        )

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

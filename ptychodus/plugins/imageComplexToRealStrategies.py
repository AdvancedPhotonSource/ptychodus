import colorsys

import numpy

from ptychodus.api.image import ComplexArrayType, ComplexToRealStrategy, \
        RealArrayType, ScalarTransformation
from ptychodus.api.plugins import PluginRegistry


class ComplexHSVMagnitudeInSaturation(ComplexToRealStrategy):

    @property
    def name(self) -> str:
        return 'HSV Saturation'

    @property
    def isCyclic(self) -> bool:
        return False

    @property
    def isColorized(self) -> bool:
        return True

    def __call__(self, array: ComplexArrayType,
                 transformation: ScalarTransformation) -> RealArrayType:
        f = numpy.vectorize(colorsys.hsv_to_rgb)
        h = (numpy.angle(array) + numpy.pi) / (2. * numpy.pi)
        s = transformation(numpy.absolute(array))
        v = numpy.zeros_like(h)
        return numpy.stack(f(h, s, v), axis=-1)


class ComplexHSVMagnitudeInValue(ComplexToRealStrategy):

    @property
    def name(self) -> str:
        return 'HSV Value'

    @property
    def isCyclic(self) -> bool:
        return False

    @property
    def isColorized(self) -> bool:
        return True

    def __call__(self, array: ComplexArrayType,
                 transformation: ScalarTransformation) -> RealArrayType:
        f = numpy.vectorize(colorsys.hsv_to_rgb)
        h = (numpy.angle(array) + numpy.pi) / (2. * numpy.pi)
        s = numpy.zeros_like(h)
        v = transformation(numpy.absolute(array))
        return numpy.stack(f(h, s, v), axis=-1)


class ComplexHLSMagnitudeInLightness(ComplexToRealStrategy):

    @property
    def name(self) -> str:
        return 'HLS Lightness'

    @property
    def isCyclic(self) -> bool:
        return False

    @property
    def isColorized(self) -> bool:
        return True

    def __call__(self, array: ComplexArrayType,
                 transformation: ScalarTransformation) -> RealArrayType:
        f = numpy.vectorize(colorsys.hls_to_rgb)
        h = (numpy.angle(array) + numpy.pi) / (2. * numpy.pi)
        l = transformation(numpy.absolute(array))
        s = numpy.zeros_like(h)
        return numpy.stack(f(h, l, s), axis=-1)


class ComplexHLSMagnitudeInSaturation(ComplexToRealStrategy):

    @property
    def name(self) -> str:
        return 'HLS Saturation'

    @property
    def isCyclic(self) -> bool:
        return False

    @property
    def isColorized(self) -> bool:
        return True

    def __call__(self, array: ComplexArrayType,
                 transformation: ScalarTransformation) -> RealArrayType:
        f = numpy.vectorize(colorsys.hls_to_rgb)
        h = (numpy.angle(array) + numpy.pi) / (2. * numpy.pi)
        l = numpy.zeros_like(h)
        s = transformation(numpy.absolute(array))
        return numpy.stack(f(h, l, s), axis=-1)


class ComplexMagnitudeStrategy(ComplexToRealStrategy):

    @property
    def name(self) -> str:
        return 'Magnitude'

    @property
    def isCyclic(self) -> bool:
        return False

    @property
    def isColorized(self) -> bool:
        return False

    def __call__(self, array: ComplexArrayType,
                 transformation: ScalarTransformation) -> RealArrayType:
        values = numpy.absolute(array)
        return transformation(values)


class ComplexPhaseStrategy(ComplexToRealStrategy):

    @property
    def name(self) -> str:
        return 'Phase'

    @property
    def isCyclic(self) -> bool:
        return True

    @property
    def isColorized(self) -> bool:
        return False

    def __call__(self, array: ComplexArrayType,
                 transformation: ScalarTransformation) -> RealArrayType:
        values = numpy.angle(array)
        return transformation(values)


class ComplexRealComponentStrategy(ComplexToRealStrategy):

    @property
    def name(self) -> str:
        return 'Real'

    @property
    def isCyclic(self) -> bool:
        return False

    @property
    def isColorized(self) -> bool:
        return False

    def __call__(self, array: ComplexArrayType,
                 transformation: ScalarTransformation) -> RealArrayType:
        values = numpy.real(array)
        return transformation(values)


class ComplexImaginaryComponentStrategy(ComplexToRealStrategy):

    @property
    def name(self) -> str:
        return 'Imaginary'

    @property
    def isCyclic(self) -> bool:
        return False

    @property
    def isColorized(self) -> bool:
        return False

    def __call__(self, array: ComplexArrayType,
                 transformation: ScalarTransformation) -> RealArrayType:
        values = numpy.imag(array)
        return transformation(values)


def registerPlugins(registry: PluginRegistry) -> None:
    registry.registerPlugin(ComplexMagnitudeStrategy())
    registry.registerPlugin(ComplexPhaseStrategy())
    registry.registerPlugin(ComplexRealComponentStrategy())
    registry.registerPlugin(ComplexImaginaryComponentStrategy())
    # FIXME registry.registerPlugin(ComplexHSVMagnitudeInSaturation())
    # FIXME registry.registerPlugin(ComplexHSVMagnitudeInValue())
    # FIXME registry.registerPlugin(ComplexHLSMagnitudeInLightness())
    # FIXME registry.registerPlugin(ComplexHLSMagnitudeInSaturation())

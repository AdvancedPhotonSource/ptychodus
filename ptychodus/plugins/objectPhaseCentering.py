import numpy

from ptychodus.api.object import ObjectArrayType, ObjectPhaseCenteringStrategy
from ptychodus.api.plugins import PluginRegistry


class IdentityPhaseCenteringStrategy(ObjectPhaseCenteringStrategy):

    @property
    def name(self) -> str:
        return 'Identity'

    def __call__(self, array: ObjectArrayType) -> ObjectArrayType:
        return array


class CenterBoxMeanPhaseCenteringStrategy(ObjectPhaseCenteringStrategy):

    @property
    def name(self) -> str:
        return 'CenterBoxMean'

    def __call__(self, array: ObjectArrayType) -> ObjectArrayType:
        oneThirdHeight = array.shape[-2] // 3
        oneThirdWidth = array.shape[-1] // 3

        amplitude = numpy.absolute(array)
        phase = numpy.angle(array)

        centerBoxMeanPhase = phase[oneThirdHeight:oneThirdHeight * 2,
                                   oneThirdWidth:oneThirdWidth * 2].mean()

        return amplitude * numpy.exp(1j * (phase - centerBoxMeanPhase))


def registerPlugins(registry: PluginRegistry) -> None:
    registry.registerPlugin(IdentityPhaseCenteringStrategy())
    registry.registerPlugin(CenterBoxMeanPhaseCenteringStrategy())

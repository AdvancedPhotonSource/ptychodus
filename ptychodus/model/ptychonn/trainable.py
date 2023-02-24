from abc import abstractmethod

import numpy
import numpy.typing

from ...api.reconstructor import Reconstructor

TrainingData = numpy.typing.NDArray[numpy.float32]


class TrainableReconstructor(Reconstructor):

    @abstractmethod
    def train(self, diffractionPatterns: TrainingData, reconstructedPatches: TrainingData) -> None:
        pass

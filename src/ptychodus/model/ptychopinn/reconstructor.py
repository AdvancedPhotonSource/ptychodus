from __future__ import annotations
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
import logging

import numpy
import numpy.typing

from ptycho.loader import PtychoDataContainer

from ptychodus.api.reconstructor import (
    ReconstructInput,
    ReconstructOutput,
    TrainableReconstructor,
    TrainOutput,
)

from .settings import (
    PtychoPINNInferenceSettings,
    PtychoPINNModelSettings,
    PtychoPINNTrainingSettings,
)

__all__ = [
    'PtychoPINNTrainableReconstructor',
]

logger = logging.getLogger(__name__)


def createPtychoDataContainer(parameters: ReconstructInput) -> PtychoDataContainer:
    diff3d = parameters.patterns
    probeGuess = parameters.product.probe.getIncoherentMode(0)
    objectGuess = parameters.product.object_.getLayer(0)

    logger.debug('createPtychoDataContainer pre-condition sizes:')
    logger.debug(f'diffractionPatterns: {diff3d.shape}')
    logger.debug(f'probeGuess: {probeGuess.shape}')
    logger.debug(f'objectGuess: {objectGuess.shape}')
    logger.debug(f'scanCoordinates: {len(parameters.product.scan)}')

    if objectGuess is not None:
        objectGuess = objectGuess[0, :, :]

    return PtychoDataContainer.from_raw_data_without_pc(
        xcoords=numpy.array([p.positionXInMeters for p in parameters.product.scan]),
        ycoords=numpy.array([p.positionYInMeters for p in parameters.product.scan]),
        diff3d=diff3d,
        probeGuess=probeGuess,
        # Assuming all patches are from the same object
        scan_index=numpy.zeros(len(diff3d)),  # FIXME
        objectGuess=objectGuess,
    )


class PtychoPINNTrainableReconstructor(TrainableReconstructor):
    def __init__(
        self,
        name: str,
        model_settings: PtychoPINNModelSettings,
        training_settings: PtychoPINNTrainingSettings,
        inference_settings: PtychoPINNInferenceSettings,
    ) -> None:
        super().__init__()
        self._name = name
        self._model_settings = model_settings
        self._training_settings = training_settings
        self._inference_settings = inference_settings

        # Note the model parameter 'N' is the diffraction pattern size in pixels (power of two)
        # FIXME model path to/from settings

        ptychopinnVersion = version('ptychopinn')
        logger.info(f'\tPtychoPINN {ptychopinnVersion}')

    @property
    def name(self) -> str:
        return self._name

    def reconstruct(self, parameters: ReconstructInput) -> ReconstructOutput:
        return ReconstructOutput(parameters.product, 0)  # TODO

    def ingestTrainingData(self, parameters: ReconstructInput) -> None:
        pass  # TODO

    def getOpenTrainingDataFileFilterList(self) -> Sequence[str]:
        return list()  # TODO

    def getOpenTrainingDataFileFilter(self) -> str:
        return str()  # TODO

    def openTrainingData(self, filePath: Path) -> None:
        pass  # TODO

    def getSaveTrainingDataFileFilterList(self) -> Sequence[str]:
        return list()  # TODO

    def getSaveTrainingDataFileFilter(self) -> str:
        return str()  # TODO

    def saveTrainingData(self, filePath: Path) -> None:
        pass  # TODO

    def train(self) -> TrainOutput:
        return TrainOutput([], [], 0)  # TODO

    def clearTrainingData(self) -> None:
        pass  # TODO

    def getOpenModelFileFilterList(self) -> Sequence[str]:
        return list()  # TODO

    def getOpenModelFileFilter(self) -> str:
        return str()  # TODO

    def openModel(self, filePath: Path) -> None:
        pass  # TODO

    def getSaveModelFileFilterList(self) -> Sequence[str]:
        return list()  # TODO

    def getSaveModelFileFilter(self) -> str:
        return str()  # TODO

    def saveModel(self, filePath: Path) -> None:
        pass  # TODO

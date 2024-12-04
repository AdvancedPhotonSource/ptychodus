from __future__ import annotations
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
import logging

import numpy
import numpy.typing

from ptycho import train_pinn
from ptycho.loader import PtychoDataContainer

from ptychodus.api.product import Product
from ptychodus.api.reconstructor import (
    ReconstructInput,
    ReconstructOutput,
    TrainableReconstructor,
    TrainOutput,
)

from ..analysis import ObjectStitcher
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
    probeGuess = parameters.product.probe.array[0, :, :]
    objectGuess = parameters.product.object_.array[0, :, :]

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
        model_settings: PtychoPINNModelSettings,
        training_settings: PtychoPINNTrainingSettings,
        inference_settings: PtychoPINNInferenceSettings,
    ) -> None:
        super().__init__()
        self._model_settings = model_settings
        self._training_settings = training_settings
        self._trainingDataFileFilterList: list[str] = ['NumPy Zipped Archive (*.npz)']
        self._modelFileFilterList: list[str] = list()  # FIXME

        self._trainingData: ReconstructInput | None = None
        # FIXME self._modelInstance = modelInstance
        # FIXME self._history = history

        ptychopinnVersion = version('ptychopinn')
        logger.info(f'\tPtychoPINN {ptychopinnVersion}')

    @property
    def name(self) -> str:
        return 'PtychoPINN'

    def reconstruct(self, parameters: ReconstructInput) -> ReconstructOutput:
        if self._history is None:
            raise ValueError('No history!')

        if self._modelInstance is None:
            raise ValueError('No model instance!')

        testData = createPtychoDataContainer(parameters)
        evalResults = train_pinn.eval(testData, self._history, self._modelInstance)
        objectPatches = evalResults['reconstructed_obj'][:, :, :, 0]

        logger.debug('Stitching...')
        stitcher = ObjectStitcher(parameters.product.object_.getGeometry())

        for scanPoint, patchArray in zip(parameters.product.scan, objectPatches):
            stitcher.addPatch(scanPoint, patchArray)

        product = Product(
            metadata=parameters.product.metadata,
            scan=parameters.product.scan,
            probe=parameters.product.probe,
            object_=stitcher.build(),
            costs=list(),  # TODO put something here?
        )

        return ReconstructOutput(product, 0)

    def ingestTrainingData(self, parameters: ReconstructInput) -> None:
        self._trainingData = parameters

    def getOpenTrainingDataFileFilterList(self) -> Sequence[str]:
        return self._trainingDataFileFilterList

    def getOpenTrainingDataFileFilter(self) -> str:
        return self._trainingDataFileFilterList[0]

    def openTrainingData(self, filePath: Path) -> None:
        raise NotImplementedError(f'Open training data from "{filePath}"')  # FIXME

    def getSaveTrainingDataFileFilterList(self) -> Sequence[str]:
        return self._trainingDataFileFilterList

    def getSaveTrainingDataFileFilter(self) -> str:
        return self._trainingDataFileFilterList[0]

    def saveTrainingData(self, filePath: Path) -> None:
        raise NotImplementedError(f'Save training data to "{filePath}"')  # FIXME

    def train(self) -> TrainOutput:
        parameters = self._trainingData

        if parameters is None:
            raise ValueError('No training dataset!')

        ptychoDataContainer = createPtychoDataContainer(parameters)
        intensity_scale = train_pinn.calculate_intensity_scale(ptychoDataContainer)
        modelInstance, history = train_pinn.train(ptychoDataContainer, intensity_scale)
        self._modelInstance = modelInstance  # FIXME
        self._history = history  # FIXME

        return TrainOutput(
            trainingLoss=history.history['loss'],
            validationLoss=history.history['val_loss'],  # TODO replace with actual values
            result=0,
        )

    def clearTrainingData(self) -> None:
        self._trainingData = None

    def getOpenModelFileFilterList(self) -> Sequence[str]:
        return self._modelFileFilterList

    def getOpenModelFileFilter(self) -> str:
        return self._modelFileFilterList[0]

    def openModel(self, filePath: Path) -> None:
        raise NotImplementedError(f'Open model to "{filePath}"')  # FIXME

    def getSaveModelFileFilterList(self) -> Sequence[str]:
        return self._modelFileFilterList

    def getSaveModelFileFilter(self) -> str:
        return self._modelFileFilterList[0]

    def saveModel(self, filePath: Path) -> None:
        raise NotImplementedError(f'Save trained model to "{filePath}"')  # FIXME

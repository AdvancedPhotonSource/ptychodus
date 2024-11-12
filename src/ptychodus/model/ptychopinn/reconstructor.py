from __future__ import annotations
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path
from typing import Any
import logging

import numpy
import numpy.typing

from ptycho import params as ptycho_params, train_pinn
from ptycho.loader import PtychoDataContainer

from ptychodus.api.product import Product
from ptychodus.api.reconstructor import (
    ReconstructInput,
    ReconstructOutput,
    TrainableReconstructor,
    TrainOutput,
)

from ..analysis import ObjectStitcher
from .settings import PtychoPINNModelSettings, PtychoPINNTrainingSettings

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
        self, model_settings: PtychoPINNModelSettings, training_settings: PtychoPINNTrainingSettings
    ) -> None:
        super().__init__()
        self._model_settings = model_settings
        self._training_settings = training_settings
        self._trainingDataFileFilterList: list[str] = ['NumPy Zipped Archive (*.npz)']
        self._modelFileFilterList: list[str] = list()  # TODO

        self._trainingData: ReconstructInput | None = None
        self._modelInstance = modelInstance  # FIXME
        self._history = history  # FIXME

        ptychopinnVersion = version('ptychopinn')
        logger.info(f'\tPtychoPINN {ptychopinnVersion}')

        self._syncSettingsToPtycho(nphotons=-1)

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
        self._syncSettingsToPtycho(nphotons=int(parameters.product.metadata.probePhotons))
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

    def _syncSettingsToPtycho(self, *, nphotons: int) -> None:
        N = self._model_settings.N.getValue()
        gridsize = self._model_settings.gridsize.getValue()
        offset = self._model_settings.offset.getValue()
        bigN = N + (gridsize - 1) * offset
        max_position_jitter = 10
        buffer = max_position_jitter

        cfg: dict[str, Any] = {
            'learning_rate': self._model_settings.learning_rate.getValue(),
            'N': N,
            'offset': offset,
            'gridsize': gridsize,
            'outer_offset_train': None,
            'outer_offset_test': None,
            'batch_size': self._model_settings.batch_size.getValue(),
            'nepochs': 60,
            'n_filters_scale': self._model_settings.n_filters_scale.getValue(),
            'output_path': self._training_settings.output_path.getValue(),
            'output_prefix': self._training_settings.output_prefix.getValue(),
            'output_suffix': self._training_settings.output_suffix.getValue(),
            'big_gridsize': 10,
            'max_position_jitter': max_position_jitter,
            'sim_jitter_scale': 0.0,
            'default_probe_scale': 0.7,
            'mae_weight': self._training_settings.mae_weight.getValue(),
            'nll_weight': self._training_settings.nll_weight.getValue(),
            'tv_weight': self._training_settings.tv_weight.getValue(),
            'realspace_mae_weight': self._training_settings.realspace_mae_weight.getValue(),
            'realspace_weight': self._training_settings.realspace_weight.getValue(),
            'nimgs_train': 9,
            'nimgs_test': 3,
            'data_source': 'generic',
            'probe.trainable': self._model_settings.is_probe_trainable.getValue(),
            'intensity_scale.trainable': self._model_settings.intensity_scale_trainable.getValue(),
            'positions.provided': False,
            'object.big': self._model_settings.object_big.getValue(),
            'probe.big': self._model_settings.probe_big.getValue(),  # if True, increase the real space solution from 32x32 to 64x64
            'probe_scale': self._model_settings.probe_scale.getValue(),
            'set_phi': False,
            'probe.mask': self._model_settings.probe_mask.getValue(),
            'model_type': self._model_settings.model_type.getValue(),
            'label': '',
            'size': self._model_settings.size.getValue(),
            'amp_activation': self._model_settings.amp_activation.getValue(),
            # derived values
            'bigN': bigN,
            'padded_size': bigN + buffer,
            'padding_size': (gridsize - 1) * offset + buffer,
        }

        if nphotons > 0:
            cfg['nphotons'] = nphotons

        # sync current settings to ptycho's configuration
        ptycho_params.cfg.update(cfg)

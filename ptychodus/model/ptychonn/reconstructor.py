from importlib.metadata import version
from pathlib import Path
from typing import Optional
import logging

import numpy
import numpy.typing

import ptychonn
from ptychonn import ReconSmallPhaseModel, Tester, Trainer

from ...api.reconstructor import ReconstructResult, Reconstructor
from ...api.scan import Scan
from ..data import ActiveDiffractionDataset
from ..object import Object
from ..probe import Apparatus
from .settings import PtychoNNModelSettings, PtychoNNTrainingSettings

logger = logging.getLogger(__name__)


class PtychoNNPhaseOnlyReconstructor(Reconstructor):

    def __init__(self, settings: PtychoNNModelSettings, trainingSettings: PtychoNNTrainingSettings,
                 apparatus: Apparatus, scan: Scan, object_: Object,
                 diffractionDataset: ActiveDiffractionDataset) -> None:
        self._settings = settings
        self._trainingSettings = trainingSettings
        self._apparatus = apparatus
        self._scan = scan
        self._object = object_
        self._diffractionDataset = diffractionDataset

        ptychonnVersion = version('ptychonn')
        logger.info(f'\tPtychoNN {ptychonnVersion}')

    @property
    def name(self) -> str:
        return 'PhaseOnly'

    def _createModel(self) -> ReconSmallPhaseModel:
        logger.debug('Building model...')
        return ReconSmallPhaseModel(
            nconv=self._settings.numberOfConvolutionChannels.value,
            use_batch_norm=self._settings.useBatchNormalization.value,
        )

    def reconstruct(self) -> ReconstructResult:
        assembledIndexes = self._diffractionDataset.getAssembledIndexes()

        logger.debug('Preparing scan data...')

        scanXInMeters: list[float] = list()
        scanYInMeters: list[float] = list()

        for index in assembledIndexes:
            try:
                point = self._scan[index]
            except KeyError:
                continue

            scanXInMeters.append(float(point.x))
            scanYInMeters.append(float(point.y))

        scanInMeters = numpy.column_stack((scanYInMeters, scanXInMeters)).astype('float32')

        data = self._diffractionDataset.getAssembledData()
        dataSize = data.shape[-1]

        if dataSize != data.shape[-2]:
            raise ValueError('PtychoNN expects square diffraction data!')

        isDataSizePow2 = (dataSize & (dataSize - 1) == 0 and dataSize > 0)

        if not isDataSizePow2:
            raise ValueError('PtychoNN expects that the diffraction data size is a power of two!')

        # Bin diffraction data
        inputSize = self._settings.inputSize.value
        binSize = dataSize // inputSize

        if binSize == 1:
            binnedData = data
        else:
            binnedData = numpy.zeros((data.shape[0], inputSize, inputSize), dtype=data.dtype)

            for i in range(inputSize):
                for j in range(inputSize):
                    binnedData[:, i, j] = numpy.sum(data[:, binSize * i:binSize * (i + 1),
                                                         binSize * j:binSize * (j + 1)])

        stitchedPixelWidthInMeters = self._apparatus.getObjectPlanePixelSizeXInMeters()
        inferencePixelWidthInMeters = stitchedPixelWidthInMeters * binSize

        logger.debug('Loading model state...')
        tester = Tester(
            model=self._createModel(),
            model_params_path=self._settings.stateFilePath.value,
        )

        logger.debug('Inferring...')
        tester.setTestData(binnedData, batch_size=self._settings.batchSize.value)
        inferences = tester.predictTestData()

        logger.debug('Stitching...')
        stitchedPhase = ptychonn.stitch_from_inference(
            inferences,
            scanInMeters,
            stitched_pixel_width=float(stitchedPixelWidthInMeters),
            inference_pixel_width=float(inferencePixelWidthInMeters))
        stitched = numpy.exp(1j * stitchedPhase)
        self._object.setArray(stitched)

        return ReconstructResult(0, [[]])

    def train(self, diffractionPatterns: numpy.typing.NDArray[numpy.float32],
              reconstructedPatches: numpy.typing.NDArray[numpy.float32]) -> None:
        outputPath = self._trainingSettings.outputPath.value \
                if self._trainingSettings.saveTrainingArtifacts.value else None
        trainer = Trainer(
            model=self._createModel(),
            batch_size=self._settings.batchSize.value,
            output_path=outputPath,
            output_suffix=self._trainingSettings.outputSuffix.value,
        )
        trainer.setTrainingData(
            X_train_full=diffractionPatterns,
            Y_ph_train_full=reconstructedPatches,
            valid_data_ratio=float(self._trainingSettings.validationSetFractionalSize.value),
        )
        trainer.setOptimizationParams(
            epochs_per_half_cycle=self._trainingSettings.optimizationEpochsPerHalfCycle.value,
            max_lr=float(self._trainingSettings.maximumLearningRate.value),
            min_lr=float(self._trainingSettings.minimumLearningRate.value),
        )

        logger.debug('Loading model state...')
        trainer.initModel(model_params_path=self._settings.stateFilePath.value)

        logger.debug('Training...')
        trainer.run(
            epochs=self._trainingSettings.trainingEpochs.value,
            output_frequency=self._trainingSettings.statusIntervalInEpochs.value,
        )

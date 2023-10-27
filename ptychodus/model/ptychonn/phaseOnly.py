from __future__ import annotations
from importlib.metadata import version
from pathlib import Path
from typing import Any, Mapping, TypeAlias
import logging

from ptychonn import ReconSmallModel, Tester, Trainer
from scipy.ndimage import map_coordinates
import numpy
import numpy.typing

from ...api.image import ImageExtent
from ...api.object import ObjectPatchAxis
from ...api.plot import Plot2D, PlotAxis, PlotSeries
from ...api.reconstructor import ReconstructInput, ReconstructOutput, TrainableReconstructor
from ..object import ObjectAPI
from .settings import PtychoNNModelSettings, PtychoNNTrainingSettings

FloatArrayType: TypeAlias = numpy.typing.NDArray[numpy.float32]

logger = logging.getLogger(__name__)


class CircularBuffer:

    def __init__(self, extent: ImageExtent, maxSize: int) -> None:
        self._buffer: FloatArrayType = numpy.zeros((maxSize, *extent.shape), dtype=numpy.float32)
        self._pos = 0
        self._full = False

    @classmethod
    def createZeroSized(cls) -> CircularBuffer:
        return cls(ImageExtent(0, 0), 0)

    @property
    def isZeroSized(self) -> bool:
        return (self._buffer.size == 0)

    def append(self, array: FloatArrayType) -> None:
        self._buffer[self._pos, :, :] = array
        self._pos += 1

        if self._pos == self._buffer.shape[0]:
            self._pos = 0
            self._full = True

    def getBuffer(self) -> FloatArrayType:
        return self._buffer if self._full else self._buffer[:self._pos]


class PtychoNNPhaseOnlyTrainableReconstructor(TrainableReconstructor):

    def __init__(self, settings: PtychoNNModelSettings, trainingSettings: PtychoNNTrainingSettings,
                 objectAPI: ObjectAPI) -> None:
        self._settings = settings
        self._trainingSettings = trainingSettings
        self._objectAPI = objectAPI
        self._diffractionPatternBuffer = CircularBuffer.createZeroSized()
        self._objectPatchBuffer = CircularBuffer.createZeroSized()

        ptychonnVersion = version('ptychonn')
        logger.info(f'\tPtychoNN {ptychonnVersion}')

    @property
    def name(self) -> str:
        return 'PhaseOnly'

    def _createModel(self) -> ReconSmallModel:
        logger.debug('Building model...')
        return ReconSmallModel(
            nconv=self._settings.numberOfConvolutionChannels.value,
            use_batch_norm=self._settings.useBatchNormalization.value,
        )

    def reconstruct(self, parameters: ReconstructInput) -> ReconstructOutput:
        # TODO data size/shape requirements to GUI
        data = parameters.diffractionPatternArray
        dataSize = data.shape[-1]

        if dataSize != data.shape[-2]:
            raise ValueError('PtychoNN expects square diffraction data!')

        isDataSizePow2 = (dataSize & (dataSize - 1) == 0 and dataSize > 0)

        if not isDataSizePow2:
            raise ValueError('PtychoNN expects that the diffraction data size is a power of two!')

        # Bin diffraction data
        # TODO extract binning to data loading (and verify that x-y coordinates are correct)
        inputSize = dataSize
        binSize = dataSize // inputSize

        if binSize == 1:
            binnedData = data
        else:
            binnedData = numpy.zeros((data.shape[0], inputSize, inputSize), dtype=data.dtype)

            for i in range(inputSize):
                for j in range(inputSize):
                    binnedData[:, i, j] = numpy.sum(data[:, binSize * i:binSize * (i + 1),
                                                         binSize * j:binSize * (j + 1)])

        logger.debug('Loading model state...')
        tester = Tester(
            model=self._createModel(),
            model_params_path=self._settings.stateFilePath.value,
        )

        logger.debug('Inferring...')
        tester.setTestData(binnedData.astype(numpy.float32),
                           batch_size=self._settings.batchSize.value)
        npzSavePath = None  # TODO self._trainingSettings.outputPath.value / 'preds.npz'
        objectPatches = tester.predictTestData(npz_save_path=npzSavePath)

        logger.debug('Stitching...')
        objectInterpolator = parameters.objectInterpolator
        objectGrid = objectInterpolator.getGrid()
        objectArray = objectInterpolator.getArray()
        objectArrayUpper = numpy.zeros_like(objectArray, dtype=complex)
        objectArrayCount = numpy.zeros_like(objectArray, dtype=float)

        patchExtent = ImageExtent(
            width=objectPatches.shape[-1],
            height=objectPatches.shape[-2],
        )

        for scanPoint, objectPatchReals in zip(parameters.scan.values(), objectPatches):
            objectPatch = 0.5 * numpy.exp(1j * objectPatchReals[0])

            patchAxisX = ObjectPatchAxis(objectGrid.axisX, scanPoint.x, patchExtent.width)
            patchAxisY = ObjectPatchAxis(objectGrid.axisY, scanPoint.y, patchExtent.height)

            pixelCentersX = patchAxisX.getObjectPixelCenters()
            pixelCentersY = patchAxisY.getObjectPixelCenters()

            xx, yy = numpy.meshgrid(pixelCentersX.patchCoordinates, pixelCentersY.patchCoordinates)
            patchValues = map_coordinates(objectPatch, (yy, xx), order=1)

            # TODO consider inverse distance weighting
            objectArrayUpper[pixelCentersY.objectSlice, pixelCentersX.objectSlice] += patchValues
            objectArrayCount[pixelCentersY.objectSlice, pixelCentersX.objectSlice] += 1

        objectArrayLower = numpy.maximum(objectArrayCount, 1)
        objectArray = objectArrayUpper / objectArrayLower

        return ReconstructOutput(
            scan=None,
            probeArray=None,
            objectArray=objectArray,
            objective=[[]],
            plot2D=Plot2D.createNull(),  # TODO show something here?
            result=0,
        )

    def ingest(self, parameters: ReconstructInput) -> None:
        objectInterpolator = parameters.objectInterpolator

        if self._diffractionPatternBuffer.isZeroSized:
            diffractionPatternExtent = parameters.diffractionPatternExtent
            maximumSize = max(1, self._trainingSettings.maximumTrainingDatasetSize.value)

            self._diffractionPatternBuffer = CircularBuffer(diffractionPatternExtent, maximumSize)
            self._objectPatchBuffer = CircularBuffer(diffractionPatternExtent, maximumSize)

        for scanIndex, scanPoint in parameters.scan.items():
            objectPatch = objectInterpolator.getPatch(scanPoint, parameters.probeExtent)
            objectPhasePatch = numpy.angle(objectPatch.array).astype(numpy.float32)
            self._objectPatchBuffer.append(objectPhasePatch)

        for pattern in parameters.diffractionPatternArray.astype(numpy.float32):
            self._diffractionPatternBuffer.append(pattern)

    def _plotMetrics(self, metrics: Mapping[str, Any]) -> Plot2D:
        trainingLoss = [losses[0] for losses in metrics['losses']]
        validationLoss = [losses[0] for losses in metrics['val_losses']]
        validationLossSeries = PlotSeries(label='Validation Loss', values=validationLoss)
        trainingLossSeries = PlotSeries(label='Training Loss', values=trainingLoss)
        seriesX = PlotSeries(label='Iteration', values=[*range(len(trainingLoss))])

        return Plot2D(
            axisX=PlotAxis(label='Epoch', series=[seriesX]),
            axisY=PlotAxis(label='Loss', series=[trainingLossSeries, validationLossSeries]),
        )

    def train(self) -> Plot2D:
        outputPath = self._trainingSettings.outputPath.value \
                if self._trainingSettings.saveTrainingArtifacts.value else None

        trainer = Trainer(
            model=self._createModel(),
            batch_size=self._settings.batchSize.value,
            output_path=outputPath,
            output_suffix=self._trainingSettings.outputSuffix.value,
        )

        X_train_full = self._diffractionPatternBuffer.getBuffer()
        Y_ph_train_full = numpy.expand_dims(self._objectPatchBuffer.getBuffer(), 1)

        trainer.setTrainingData(
            X_train_full=X_train_full,
            Y_ph_train_full=Y_ph_train_full,
            valid_data_ratio=float(self._trainingSettings.validationSetFractionalSize.value),
        )
        trainer.setOptimizationParams(
            epochs_per_half_cycle=self._trainingSettings.optimizationEpochsPerHalfCycle.value,
            max_lr=float(self._trainingSettings.maximumLearningRate.value),
            min_lr=float(self._trainingSettings.minimumLearningRate.value),
        )

        logger.debug('Loading model state...')
        trainer.initModel()

        logger.debug('Training...')
        trainer.run(
            epochs=self._trainingSettings.trainingEpochs.value,
            output_frequency=self._trainingSettings.statusIntervalInEpochs.value,
        )

        return self._plotMetrics(trainer.metrics)

    def reset(self) -> None:
        self._diffractionPatternBuffer = CircularBuffer.createZeroSized()
        self._objectPatchBuffer = CircularBuffer.createZeroSized()

    def saveTrainingData(self, filePath: Path) -> None:
        trainingData = {
            'diffractionPatterns': self._diffractionPatternBuffer.getBuffer(),
            'objectPatches': self._objectPatchBuffer.getBuffer(),
        }
        numpy.savez(filePath, **trainingData)

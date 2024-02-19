from __future__ import annotations
from collections.abc import Sequence
from ..object import ObjectAPI
from pathlib import Path
from importlib.metadata import version
from .settings import PtychoPINNModelSettings, PtychoPINNTrainingSettings
from ...api.image import ImageExtent
from ...api.object import ObjectArrayType, ObjectPatchAxis
from ...api.plot import Plot2D, PlotAxis, PlotSeries
from ...api.reconstructor import ReconstructInput, ReconstructOutput, TrainableReconstructor
import numpy
import numpy.typing
import logging

FloatArrayType = numpy.typing.NDArray[numpy.float32]
logger = logging.getLogger(__name__)

class PatternCircularBuffer:

    def __init__(self, extent: ImageExtent, maxSize: int) -> None:
        self._buffer: FloatArrayType = numpy.zeros(
            (maxSize, *extent.shape),
            dtype=numpy.float32,
        )
        self._pos = 0
        self._full = False

    @classmethod
    def createZeroSized(cls) -> PatternCircularBuffer:
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

class ObjectPatchCircularBuffer:

    def __init__(self, extent: ImageExtent, channels: int, maxSize: int) -> None:
        self._buffer: FloatArrayType = numpy.zeros(
            (maxSize, channels, *extent.shape),
            dtype=numpy.float32,
        )
        self._pos = 0
        self._full = False

    @classmethod
    def createZeroSized(cls) -> ObjectPatchCircularBuffer:
        return cls(ImageExtent(0, 0), 0, 0)

    @property
    def isZeroSized(self) -> bool:
        return (self._buffer.size == 0)

    def append(self, array: ObjectArrayType) -> None:
        self._buffer[self._pos, 0, :, :] = numpy.angle(array).astype(numpy.float32)

        if self._buffer.shape[1] > 1:
            self._buffer[self._pos, 1, :, :] = numpy.absolute(array).astype(numpy.float32)

        self._pos += 1

        if self._pos == self._buffer.shape[0]:
            self._pos = 0
            self._full = True

    def getBuffer(self) -> FloatArrayType:
        return self._buffer if self._full else self._buffer[:self._pos]
class PtychoPINNTrainableReconstructor(TrainableReconstructor):

    def __init__(self, modelSettings: PtychoPINNModelSettings, trainingSettings: PtychoPINNTrainingSettings, objectAPI: ObjectAPI, *, enableAmplitude: bool) -> None:
        self._modelSettings = modelSettings
        self._trainingSettings = trainingSettings
        self._objectAPI = objectAPI
        self._enableAmplitude = enableAmplitude
        self._fileFilterList: list[str] = ['NumPy Zipped Archive (*.npz)']
        ptychopinnVersion = version('ptychopinn')
        logger.info(f'\tPtychoPINN {ptychopinnVersion}')
        self.modelSettings = modelSettings
        self.trainingSettings = trainingSettings
        ptychopinnVersion = version('ptychopinn')
        logger.info(f'\tPtychoPINN {ptychopinnVersion}')
        self._patternBuffer = PatternCircularBuffer.createZeroSized()
        self._objectPatchBuffer = ObjectPatchCircularBuffer.createZeroSized()
        self._fileFilterList = ['NumPy Zipped Archive (*.npz)']
        self.fileFilterList = ['NumPy Arrays (*.npy)', 'NumPy Zipped Archive (*.npz)']
        self._initialize_ptycho()

    def _initialize_ptycho(self) -> None:
        from ptycho import params as ptycho_params
        from .params import update_cfg_from_settings
        from . import params
        # Adjusted to not directly access settingsRegistry from PtychoPINNModelSettings
        # Assuming an alternative method to obtain settings is used here
        # Placeholder for the correct approach to update settings
        # update_cfg_from_settings(correct_approach_to_obtain_settings)
        # Directly update ptycho's cfg dictionary based on the updated values from ptychodus
        for key, value in params.cfg.items():
            ptycho_params.cfg[key] = value


    @property
    def name(self) -> str:
        return 'AmplitudePhase' if self._enableAmplitude else 'PhaseOnly'

    # Placeholder for the reconstruct method remains as implementing the actual logic requires details about the PtychoPINN model.

    def ingestTrainingData(self, parameters: ReconstructInput) -> None:
        # Adjusted to match the API specification and example implementation. Actual logic depends on the model details.
        objectInterpolator = parameters.objectInterpolator

        if self._patternBuffer.isZeroSized:
            diffractionPatternExtent = parameters.diffractionPatternExtent
            maximumSize = max(1, self._trainingSettings.maximumTrainingDatasetSize.value)

            channels = 2 if self._enableAmplitude else 1
            self._patternBuffer = PatternCircularBuffer(diffractionPatternExtent, maximumSize)
            self._objectPatchBuffer = ObjectPatchCircularBuffer(diffractionPatternExtent, channels, maximumSize)

        for scanIndex, scanPoint in parameters.scan.items():
            objectPatch = objectInterpolator.getPatch(scanPoint, parameters.probeExtent)
            self._objectPatchBuffer.append(objectPatch.array)

        for pattern in parameters.diffractionPatternArray.astype(numpy.float32):
            self._patternBuffer.append(pattern)

    def getSaveFileFilterList(self) -> Sequence[str]:
        return self.fileFilterList

    def getSaveFileFilter(self) -> str:
        return self.fileFilterList[0]  # Default to the first option

    def saveTrainingData(self, filePath: Path) -> None:
        logger.debug(f'Writing \"{filePath}\" as \"NPZ\"')
        trainingData = {
            'diffractionPatterns': self._patternBuffer.getBuffer(),
            'objectPatches': self._objectPatchBuffer.getBuffer(),
        }
        numpy.savez(filePath, **trainingData)

    def train(self) -> Plot2D:
        # Detailed TODO: Implement the model training logic specific to PtychoPINN
        # This should include initializing the model, preparing the data, running the training loop,
        # and validating the model. The specifics of these steps depend on the PtychoPINN architecture
        # and training procedure, which are not detailed here.
        #
        # After training, generate a Plot2D object to visualize the training progress, such as loss over epochs.
        # This visualization is crucial for understanding the training dynamics and evaluating the model's performance.
        #
        # Placeholder for training logic:
        # Initialize model, prepare data, run training loop, validate model
        #
        # Placeholder for generating Plot2D object:
        trainingLoss = [0]  # Replace with actual training loss values
        validationLoss = [0]  # Replace with actual validation loss values
        validationLossSeries = PlotSeries(label='Validation Loss', values=validationLoss)
        trainingLossSeries = PlotSeries(label='Training Loss', values=trainingLoss)
        seriesX = PlotSeries(label='Epoch', values=[*range(len(trainingLoss))])

        return Plot2D(
            axisX=PlotAxis(label='Epoch', series=[seriesX]),
            axisY=PlotAxis(label='Loss', series=[trainingLossSeries, validationLossSeries]),
        )

    def clearTrainingData(self) -> None:
        logger.debug('Clearing training data...')
        self._patternBuffer = PatternCircularBuffer.createZeroSized()
        self._objectPatchBuffer = ObjectPatchCircularBuffer.createZeroSized()

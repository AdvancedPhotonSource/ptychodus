import logging

from ptychonn import Tester
from scipy.ndimage import map_coordinates
import numpy

from ...api.image import ImageExtent
from ...api.object import ObjectPatchAxis
from ...api.reconstructor import Reconstructor, ReconstructInput, ReconstructOutput
from ..object import ObjectAPI
from .factory import PtychoNNModelFactory
from .settings import PtychoNNModelSettings

logger = logging.getLogger(__name__)


class PtychoNNPhaseOnlyReconstructor(Reconstructor):

    def __init__(self, settings: PtychoNNModelSettings, factory: PtychoNNModelFactory,
                 objectAPI: ObjectAPI) -> None:
        self._settings = settings
        self._factory = factory
        self._objectAPI = objectAPI

    @property
    def name(self) -> str:
        return 'InferPhase'

    def execute(self, parameters: ReconstructInput) -> ReconstructOutput:
        data = parameters.diffractionPatternArray
        dataSize = data.shape[-1]

        if dataSize != data.shape[-2]:
            raise ValueError('PtychoNN expects square diffraction data!')

        isDataSizePow2 = (dataSize & (dataSize - 1) == 0 and dataSize > 0)

        if not isDataSizePow2:
            raise ValueError('PtychoNN expects that the diffraction data size is a power of two!')

        # Bin diffraction data
        # TODO extract binning to data loading (and verify that x-y coordinates are correct)
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

        logger.debug('Loading model state...')
        tester = Tester(
            model=self._factory.createModel(),
            model_params_path=self._settings.stateFilePath.value,
        )

        logger.debug('Inferring...')
        tester.setTestData(binnedData, batch_size=self._settings.batchSize.value)
        objectPhasePatches = tester.predictTestData()

        logger.debug('Stitching...')
        objectInterpolator = parameters.objectInterpolator
        objectGrid = objectInterpolator.getGrid()
        objectArray = objectInterpolator.getArray()
        objectArrayUpper = numpy.zeros_like(objectArray, dtype=complex)
        objectArrayCount = numpy.zeros_like(objectArray, dtype=float)

        patchExtent = ImageExtent(
            width=objectPhasePatches.shape[-1],
            height=objectPhasePatches.shape[-2],
        )

        for scanPoint, objectPhasePatch in zip(parameters.scan.values(), objectPhasePatches):
            objectPatch = numpy.exp(1j * objectPhasePatch)

            patchAxisX = ObjectPatchAxis(objectGrid.axisX, float(scanPoint.x), patchExtent.width)
            patchAxisY = ObjectPatchAxis(objectGrid.axisY, float(scanPoint.y), patchExtent.height)

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
            result=0,
        )

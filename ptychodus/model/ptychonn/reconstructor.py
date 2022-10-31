from importlib.metadata import version
import logging

import numpy

import ptychonn
from ptychonn._model import ReconSmallPhaseModel, Tester

from ...api.reconstructor import ReconstructResult, Reconstructor
from ..data import ActiveDiffractionDataset
from ..object import Object
from ..probe import Apparatus
from ..scan import Scan
from .settings import PtychoNNSettings

logger = logging.getLogger(__name__)


class PtychoNNReconstructor(Reconstructor):

    def __init__(self, settings: PtychoNNSettings, apparatus: Apparatus, scan: Scan,
                 object_: Object, diffractionDataset: ActiveDiffractionDataset) -> None:
        self._settings = settings
        self._apparatus = apparatus
        self._scan = scan
        self._object = object_
        self._diffractionDataset = diffractionDataset

        ptychonnVersion = version('ptychonn')
        logger.info(f'\tPtychoNN {ptychonnVersion}')

    @property
    def name(self) -> str:
        return 'PtychoNN'

    def reconstruct(self) -> ReconstructResult:
        assembledIndexes = self._diffractionDataset.getAssembledIndexes()
        scanXInMeters: list[float] = list()
        scanYInMeters: list[float] = list()

        for index in assembledIndexes:
            try:
                point = self._scan[index]
            except IndexError:
                continue

            scanXInMeters.append(float(point.x))
            scanYInMeters.append(float(point.y))

        scanInMeters = numpy.column_stack((scanYInMeters, scanXInMeters)).astype('float32')

        data = self._diffractionDataset.getAssembledData()
        dataSize = data.shape[-1]

        if dataSize != data.shape[-2]:
            raise ValueError('PtychoNN expects square diffraction data!')

        isDataSizePow2 = (dataSize & (dataSize - 1) and dataSize > 0)

        if not isDataSizePow2:
            raise ValueError('PtychoNN expects that the diffraction data size is a power of two!')

        # Bin diffraction data
        inputSize = self._settings.modelInputSize.value
        binnedData = numpy.zeros((data.shape[0], inputSize, inputSize))
        binSize = dataSize // inputSize

        for i in range(inputSize):
            for j in range(inputSize):
                binnedData[:, i, j] = numpy.sum(data[:, binSize * i:binSize * (i + 1),
                                                     binSize * j:binSize * (j + 1)])

        stitchedPixelWidthInMeters = self._apparatus.getObjectPlanePixelSizeXInMeters()
        inferencePixelWidthInMeters = stitchedPixelWidthInMeters * binSize

        # Load best_model.pth
        tester = Tester(model=ReconSmallPhaseModel(),
                        model_params_path=self._settings.modelStateFilePath.value)

        # Predict
        tester.setTestData(data, batch_size=self._settings.batchSize.value)
        inferences = tester.predictTestData()

        # Stitch
        stitched = ptychonn.stitch_from_inference(inferences, scanInMeters,
                                                  float(stitchedPixelWidthInMeters),
                                                  float(inferencePixelWidthInMeters))
        self._object.setArray(stitched)

        return ReconstructResult(0, [[]])

from importlib.metadata import version
import logging

import numpy

import ptychonn
from ptychonn import ReconSmallPhaseModel, Tester

from ...api.reconstructor import ReconstructResult, Reconstructor
from ...api.scan import Scan
from ..data import ActiveDiffractionDataset
from ..object import Object
from ..probe import Apparatus
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

        logger.debug('Validating diffraction pattern data...')

        data = self._diffractionDataset.getAssembledData()
        dataSize = data.shape[-1]

        if dataSize != data.shape[-2]:
            raise ValueError('PtychoNN expects square diffraction data!')

        isDataSizePow2 = (dataSize & (dataSize - 1) == 0 and dataSize > 0)

        if not isDataSizePow2:
            raise ValueError('PtychoNN expects that the diffraction data size is a power of two!')

        # Bin diffraction data
        inputSize = self._settings.modelInputSize.value
        binSize = dataSize // inputSize

        if binSize == 1:
            binnedData = data
        else:
            binnedData = numpy.zeros((data.shape[0], inputSize, inputSize), dtype=data.dtype)

            for i in range(inputSize):
                for j in range(inputSize):
                    binnedData[:, i, j] = numpy.sum(data[:, binSize * i:binSize * (i + 1),
                                                         binSize * j:binSize * (j + 1)])

        # used when preparing data, doesn't go into stitching code (see 99-100 in dataPrep script)
        stitchedPixelWidthInMeters = 11.176e-9  # FIXME self._apparatus.getObjectPlanePixelSizeXInMeters()
        inferencePixelWidthInMeters = 10.e-9  # FIXME stitchedPixelWidthInMeters * binSize
        # FIXME verify scan coordinate transformation

        model = ReconSmallPhaseModel(
            nconv=self._settings.numberOfConvolutionChannels.value,
            use_batch_norm=self._settings.useBatchNormalization.value,
        )
        logger.debug('Loading model state...')
        tester = Tester(model=model, model_params_path=self._settings.modelStateFilePath.value)

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

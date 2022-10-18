from importlib.metadata import version
import logging

import numpy

import ptychonn
from ptychonn._model import ReconSmallPhaseModel, Tester

from ...api.reconstructor import Reconstructor
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

    def reconstruct(self) -> int:
        stitchedPixelWidthInMeters = self._apparatus.getObjectPlanePixelSizeXInMeters()
        inferencePixelWidthInMeters = 0.  # FIXME

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

        # TODO swapXY?
        scanInMeters = numpy.column_stack((scanYInMeters, scanXInMeters)).astype('float32')

        data = self._diffractionDataset.getAssembledData()
        # TODO resize data

        # Load best_model.pth
        tester = Tester(model=ReconSmallPhaseModel(),
                        model_params_path=self._settings.modelStateFilePath.value)

        # Predict
        tester.setTestData(data, batch_size=self._settings.batchSize.value)
        inferences = tester.predictTestData()

        # Stitch
        stitched = ptychonn.stitch_from_inference(inferences, scanInMeters,
                                                  stitchedPixelWidthInMeters,
                                                  inferencePixelWidthInMeters)
        self._object.setArray(stitched)

        return 0

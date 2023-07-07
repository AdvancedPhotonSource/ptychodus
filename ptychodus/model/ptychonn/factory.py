from importlib.metadata import version
import logging

from ptychonn import ReconSmallPhaseModel

from .settings import PtychoNNModelSettings

logger = logging.getLogger(__name__)


class PtychoNNModelFactory:

    def __init__(self, settings: PtychoNNModelSettings) -> None:
        self._settings = settings

        ptychonnVersion = version('ptychonn')
        logger.info(f'\tPtychoNN {ptychonnVersion}')

    def createModel(self) -> ReconSmallPhaseModel:
        logger.debug('Building model...')
        return ReconSmallPhaseModel(
            nconv=self._settings.numberOfConvolutionChannels.value,
            use_batch_norm=self._settings.useBatchNormalization.value,
        )

from importlib.metadata import version
import logging

import ptychodus.model.ptychonn.config_temp
from .settings import PtychoNNPositionPredictionSettings

logger = logging.getLogger(__name__)

class PositionPredictionWorker:

    def __init__(self, positionPredictionSettings: PtychoNNPositionPredictionSettings) -> None:
        self._settings = positionPredictionSettings

        ptychonnVersion = version('ptychonn')
        logger.info(f'\tPtychoNN {ptychonnVersion}')

    @property
    def name(self) -> str:
        return 'PositionPOredictor'

    def run(self):
        return None
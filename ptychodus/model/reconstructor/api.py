import logging

from ...api.plot import Plot2D
from ...api.reconstructor import ReconstructOutput
from ...api.scan import ScanIndexFilter
from .active import ActiveReconstructor

logger = logging.getLogger(__name__)


class ReconstructorAPI:

    def __init__(self, activeReconstructor: ActiveReconstructor) -> None:
        super().__init__()
        self._activeReconstructor = activeReconstructor

    def reconstruct(self) -> ReconstructOutput:
        label = self._activeReconstructor.name
        result = self._activeReconstructor.reconstruct(label,
                                                       ScanIndexFilter.ALL,
                                                       selectResults=True)
        logger.info(result.result)  # TODO
        return result

    def reconstructSplit(self) -> tuple[ReconstructOutput, ReconstructOutput]:
        label = self._activeReconstructor.name
        resultOdd = self._activeReconstructor.reconstruct(f'{label} - Odd',
                                                          ScanIndexFilter.ODD,
                                                          selectResults=False)
        resultEven = self._activeReconstructor.reconstruct(f'{label} - Even',
                                                           ScanIndexFilter.EVEN,
                                                           selectResults=False)
        return resultOdd, resultEven

    def ingest(self) -> None:
        self._activeReconstructor.ingest()

    def train(self) -> Plot2D:
        return self._activeReconstructor.train()

    def reset(self) -> None:
        self._activeReconstructor.reset()

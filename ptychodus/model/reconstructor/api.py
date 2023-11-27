import logging

from ...api.visualize import Plot2D
from ...api.reconstructor import ReconstructOutput
from ...api.scan import ScanIndexFilter
from ..object.api import ObjectAPI
from .active import ActiveReconstructor

logger = logging.getLogger(__name__)


class ReconstructorAPI:

    def __init__(self, objectAPI: ObjectAPI, activeReconstructor: ActiveReconstructor) -> None:
        super().__init__()
        self._objectAPI = objectAPI
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
        labelOdd = f'{label} - Odd'
        resultOdd = self._activeReconstructor.reconstruct(labelOdd,
                                                          ScanIndexFilter.ODD,
                                                          selectResults=False)
        labelEven = f'{label} - Even'
        resultEven = self._activeReconstructor.reconstruct(labelEven,
                                                           ScanIndexFilter.EVEN,
                                                           selectResults=False)
        self._objectAPI.insertComparisonIntoRepository(labelOdd, labelEven)

        return resultOdd, resultEven

    def ingest(self) -> None:
        self._activeReconstructor.ingest()

    def train(self) -> Plot2D:
        return self._activeReconstructor.train()

    def reset(self) -> None:
        self._activeReconstructor.reset()

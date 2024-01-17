from ...api.experiment import Experiment
from ...api.scan import Scan
from ..patterns import ActiveDiffractionDataset


class ExperimentValidator:

    def __init__(self, dataset: ActiveDiffractionDataset) -> None:
        super().__init__()
        self._dataset = dataset

    def isSelectedScanValid(self, scan: Scan) -> bool:
        datasetIndexes = set(self._dataset.getAssembledIndexes())
        scanIndexes = set(point.index for point in scan)
        return (not scanIndexes.isdisjoint(datasetIndexes))

    def validate(self, experiment: Experiment) -> bool:
        # FIXME validate values (filter <0, inf, nan, etc.)
        # FIXME validate data/scan/probe/object consistency for recon
        return False  # FIXME

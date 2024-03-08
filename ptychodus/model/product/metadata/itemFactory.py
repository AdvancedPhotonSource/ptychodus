from collections.abc import Sequence
import logging

from ptychodus.api.product import ProductMetadata
from ptychodus.api.visualize import Plot2D

from ...patterns import DiffractionDatasetSettings
from ..item import ProductRepositoryItem, ProductRepositoryObserver
from ..object import ObjectRepositoryItem
from ..probe import ProbeRepositoryItem
from ..scan import ScanRepositoryItem
from .item import MetadataRepositoryItem, UniqueNameFactory

logger = logging.getLogger(__name__)


class MetadataRepositoryItemFactory(UniqueNameFactory, ProductRepositoryObserver):

    def __init__(self, repository: Sequence[ProductRepositoryItem],
                 settings: DiffractionDatasetSettings) -> None:
        self._repository = repository
        self._settings = settings
        self._reservedNames: set[str] = set()

    def create(self, metadata: ProductMetadata) -> MetadataRepositoryItem:
        return MetadataRepositoryItem(self, metadata)

    def createDefault(self, name: str, comments: str = '') -> MetadataRepositoryItem:
        metadata = ProductMetadata(
            name=name,
            comments=comments,
            probeEnergyInElectronVolts=float(self._settings.probeEnergyInElectronVolts.value),
            detectorDistanceInMeters=float(self._settings.detectorDistanceInMeters.value),
        )
        return self.create(metadata)

    def createUniqueName(self, candidateName: str) -> str:
        name = candidateName
        match = 0

        while name in self._reservedNames:
            match += 1
            name = candidateName + f'-{match}'

        return name

    def _updateLUT(self) -> None:
        self._reservedNames.clear()

        for index, item in enumerate(self._repository):
            metadata = item.getMetadata()
            metadata._index = index
            name = metadata.getName()

            if name in self._reservedNames:
                logger.warning('')  # FIXME
            else:
                self._reservedNames.add(name)

    def handleItemInserted(self, index: int, item: ProductRepositoryItem) -> None:
        self._updateLUT()

    def handleMetadataChanged(self, index: int, item: MetadataRepositoryItem) -> None:
        pass

    def handleScanChanged(self, index: int, item: ScanRepositoryItem) -> None:
        pass

    def handleProbeChanged(self, index: int, item: ProbeRepositoryItem) -> None:
        pass

    def handleObjectChanged(self, index: int, item: ObjectRepositoryItem) -> None:
        pass

    def handleCostsChanged(self, index: int, costs: Plot2D) -> None:
        pass

    def handleItemRemoved(self, index: int, item: ProductRepositoryItem) -> None:
        self._updateLUT()

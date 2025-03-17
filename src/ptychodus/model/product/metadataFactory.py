from collections.abc import Sequence
import logging

from ptychodus.api.product import ProductMetadata

from .item import ProductRepositoryItem, ProductRepositoryObserver
from .metadata import MetadataRepositoryItem, UniqueNameFactory
from .object import ObjectRepositoryItem
from .probe import ProbeRepositoryItem
from .scan import ScanRepositoryItem
from .settings import ProductSettings

logger = logging.getLogger(__name__)


class MetadataRepositoryItemFactory(UniqueNameFactory, ProductRepositoryObserver):
    def __init__(
        self, repository: Sequence[ProductRepositoryItem], settings: ProductSettings
    ) -> None:
        self._repository = repository
        self._settings = settings

    def create(self, metadata: ProductMetadata) -> MetadataRepositoryItem:
        return MetadataRepositoryItem(
            self._settings,
            self,
            name=metadata.name,
            comments=metadata.comments,
            detectorDistanceInMeters=metadata.detector_distance_m,
            probeEnergyInElectronVolts=metadata.probe_energy_eV,
            probePhotonCount=metadata.probe_photon_count,
            exposureTimeInSeconds=metadata.exposure_time_s,
        )

    def createDefault(
        self,
        *,
        name: str = '',
        comments: str = '',
        detectorDistanceInMeters: float | None = None,
        probeEnergyInElectronVolts: float | None = None,
        probePhotonCount: float | None = None,
        exposureTimeInSeconds: float | None = None,
    ) -> MetadataRepositoryItem:
        return MetadataRepositoryItem(
            self._settings,
            self,
            name=name,
            comments=comments,
            detectorDistanceInMeters=detectorDistanceInMeters,
            probeEnergyInElectronVolts=probeEnergyInElectronVolts,
            probePhotonCount=probePhotonCount,
            exposureTimeInSeconds=exposureTimeInSeconds,
        )

    def createUniqueName(self, candidateName: str) -> str:
        reservedNames = set([item.get_name() for item in self._repository])
        name = candidateName if candidateName else 'Unnamed'
        match = 0

        while name in reservedNames:
            match += 1
            name = candidateName + f'-{match}'

        return name

    def _updateLUT(self) -> None:
        for index, item in enumerate(self._repository):
            metadata = item.getMetadata()
            metadata._index = index

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

    def handleCostsChanged(self, index: int, costs: Sequence[float]) -> None:
        pass

    def handleItemRemoved(self, index: int, item: ProductRepositoryItem) -> None:
        self._updateLUT()

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
            detector_distance_m=metadata.detector_distance_m,
            probe_energy_eV=metadata.probe_energy_eV,
            probe_photon_count=metadata.probe_photon_count,
            exposure_time_s=metadata.exposure_time_s,
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
            detector_distance_m=detectorDistanceInMeters,
            probe_energy_eV=probeEnergyInElectronVolts,
            probe_photon_count=probePhotonCount,
            exposure_time_s=exposureTimeInSeconds,
        )

    def create_unique_name(self, candidateName: str) -> str:
        reservedNames = set([item.get_name() for item in self._repository])
        name = candidateName if candidateName else 'Unnamed'
        match = 0

        while name in reservedNames:
            match += 1
            name = candidateName + f'-{match}'

        return name

    def _updateLUT(self) -> None:
        for index, item in enumerate(self._repository):
            metadata = item.get_metadata()
            metadata._index = index

    def handle_item_inserted(self, index: int, item: ProductRepositoryItem) -> None:
        self._updateLUT()

    def handle_metadata_changed(self, index: int, item: MetadataRepositoryItem) -> None:
        pass

    def handle_scan_changed(self, index: int, item: ScanRepositoryItem) -> None:
        pass

    def handle_probe_changed(self, index: int, item: ProbeRepositoryItem) -> None:
        pass

    def handle_object_changed(self, index: int, item: ObjectRepositoryItem) -> None:
        pass

    def handle_costs_changed(self, index: int, costs: Sequence[float]) -> None:
        pass

    def handle_item_removed(self, index: int, item: ProductRepositoryItem) -> None:
        self._updateLUT()

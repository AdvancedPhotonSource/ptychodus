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

    def create_default(
        self,
        *,
        name: str = '',
        comments: str = '',
        detector_distance_m: float | None = None,
        probe_energy_eV: float | None = None,  # noqa: N803
        probe_photon_count: float | None = None,
        exposure_time_s: float | None = None,
    ) -> MetadataRepositoryItem:
        return MetadataRepositoryItem(
            self._settings,
            self,
            name=name,
            comments=comments,
            detector_distance_m=detector_distance_m,
            probe_energy_eV=probe_energy_eV,
            probe_photon_count=probe_photon_count,
            exposure_time_s=exposure_time_s,
        )

    def create_unique_name(self, candidate_name: str) -> str:
        reserved_names = set([item.get_name() for item in self._repository])
        name = candidate_name if candidate_name else 'Unnamed'
        match = 0

        while name in reserved_names:
            match += 1
            name = candidate_name + f'-{match}'

        return name

    def _update_lut(self) -> None:
        for index, item in enumerate(self._repository):
            metadata = item.get_metadata_item()
            metadata._index = index

    def handle_item_inserted(self, index: int, item: ProductRepositoryItem) -> None:
        self._update_lut()

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
        self._update_lut()

from ptychodus.api.observer import Observable, Observer

from ..patterns import AssembledDiffractionDataset
from .object import ObjectRepositoryItem
from .probe import ProbeRepositoryItem
from .product_geometry import ProductGeometry
from .scan import ScanRepositoryItem


class ProductValidator(Observable, Observer):
    def __init__(
        self,
        dataset: AssembledDiffractionDataset,
        scan: ScanRepositoryItem,
        geometry: ProductGeometry,
        probe: ProbeRepositoryItem,
        object_: ObjectRepositoryItem,
    ) -> None:
        super().__init__()
        self._dataset = dataset
        self._scan = scan
        self._geometry = geometry
        self._probe = probe
        self._object = object_
        self._is_scan_valid = False
        self._is_probe_valid = False
        self._is_object_valid = False

    def is_scan_valid(self) -> bool:
        return self._is_scan_valid

    def _validate_scan(self) -> None:
        scan = self._scan.get_scan()
        scan_indexes = set(point.index for point in scan)
        pattern_indexes = set(self._dataset.get_assembled_indexes())
        is_scan_valid_now = not scan_indexes.isdisjoint(pattern_indexes)

        if self._is_scan_valid != is_scan_valid_now:
            self._is_scan_valid = is_scan_valid_now
            self.notify_observers()

    def is_probe_valid(self) -> bool:
        return self._is_probe_valid

    def is_object_valid(self) -> bool:
        return self._is_object_valid

    def _validate_probe_and_object(self) -> None:
        has_validity_changed = False

        probe = self._probe.get_probe()
        is_probe_valid_now = self._geometry.is_probe_geometry_valid(probe.get_geometry())

        if self._is_probe_valid != is_probe_valid_now:
            self._is_probe_valid = is_probe_valid_now
            has_validity_changed = True

        object_ = self._object.get_object()
        is_object_valid_now = self._geometry.is_object_geometry_valid(object_.get_geometry())

        if self._is_object_valid != is_object_valid_now:
            self._is_object_valid = is_object_valid_now
            has_validity_changed = True

        if has_validity_changed:
            self.notify_observers()

    def _update(self, observable: Observable) -> None:
        if observable is self._dataset:
            self._validate_scan()
        elif observable is self._scan:
            self._validate_scan()
        elif observable is self._geometry:
            self._validate_probe_and_object()
        elif observable is self._probe:
            self._validate_probe_and_object()
        elif observable is self._object:
            self._validate_probe_and_object()

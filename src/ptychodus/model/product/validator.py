from ptychodus.api.observer import Observable, Observer

from ..diffraction import AssembledDiffractionDataset
from .geometry import ProductGeometry
from .object import ObjectRepositoryItem
from .probe import ProbeRepositoryItem
from .scan import ScanRepositoryItem


class ProductValidator(Observable, Observer):  # TODO display
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
        self._are_positions_valid = False
        self._are_probes_valid = False
        self._is_object_valid = False

    def are_positions_valid(self) -> bool:
        return self._are_positions_valid

    def are_probes_valid(self) -> bool:
        return self._are_probes_valid

    def is_object_valid(self) -> bool:
        return self._is_object_valid

    def _validate_scan(self) -> None:
        scan = self._scan.get_scan()
        scan_indexes = set(point.index for point in scan)
        pattern_indexes = set(self._dataset.get_assembled_indexes())
        are_positions_valid_now = not scan_indexes.isdisjoint(pattern_indexes)

        if self._are_positions_valid != are_positions_valid_now:
            self._are_positions_valid = are_positions_valid_now
            self.notify_observers()

    def _validate_probes_and_object(self) -> None:
        has_validity_changed = False

        probes = self._probe.get_probes()
        are_probes_valid_now = self._geometry.is_probe_geometry_valid(probes.get_geometry())

        if self._are_probes_valid != are_probes_valid_now:
            self._are_probes_valid = are_probes_valid_now
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
            self._validate_probes_and_object()
        elif observable is self._scan:
            self._validate_scan()
        elif observable is self._geometry:
            self._validate_probes_and_object()
        elif observable is self._probe:
            self._validate_probes_and_object()
        elif observable is self._object:
            self._validate_probes_and_object()

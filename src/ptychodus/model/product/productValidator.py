from ptychodus.api.observer import Observable, Observer

from ..patterns import AssembledDiffractionDataset
from .object import ObjectRepositoryItem
from .probe import ProbeRepositoryItem
from .productGeometry import ProductGeometry
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
        self._isScanValid = False
        self._isProbeValid = False
        self._isObjectValid = False

    def isScanValid(self) -> bool:
        return self._isScanValid

    def _validateScan(self) -> None:
        scan = self._scan.getScan()
        scanIndexes = set(point.index for point in scan)
        patternIndexes = set(self._dataset.get_assembled_indexes())
        isScanValidNow = not scanIndexes.isdisjoint(patternIndexes)

        if self._isScanValid != isScanValidNow:
            self._isScanValid = isScanValidNow
            self.notify_observers()

    def isProbeValid(self) -> bool:
        return self._isProbeValid

    def isObjectValid(self) -> bool:
        return self._isObjectValid

    def _validateProbeAndObject(self) -> None:
        hasValidityChanged = False

        probe = self._probe.get_probe()
        isProbeValidNow = self._geometry.isProbeGeometryValid(probe.get_geometry())

        if self._isProbeValid != isProbeValidNow:
            self._isProbeValid = isProbeValidNow
            hasValidityChanged = True

        object_ = self._object.get_object()
        isObjectValidNow = self._geometry.isObjectGeometryValid(object_.get_geometry())

        if self._isObjectValid != isObjectValidNow:
            self._isObjectValid = isObjectValidNow
            hasValidityChanged = True

        if hasValidityChanged:
            self.notify_observers()

    def _update(self, observable: Observable) -> None:
        if observable is self._dataset:
            self._validateScan()
        elif observable is self._scan:
            self._validateScan()
        elif observable is self._geometry:
            self._validateProbeAndObject()
        elif observable is self._probe:
            self._validateProbeAndObject()
        elif observable is self._object:
            self._validateProbeAndObject()

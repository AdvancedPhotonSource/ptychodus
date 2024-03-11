from __future__ import annotations
from abc import ABC, abstractmethod
import logging

from ptychodus.api.parametric import ParameterRepository
from ptychodus.api.product import ProductMetadata

logger = logging.getLogger(__name__)


class UniqueNameFactory(ABC):

    @abstractmethod
    def createUniqueName(self, candidateName: str) -> str:
        pass


class MetadataRepositoryItem(ParameterRepository):

    def __init__(self, parent: UniqueNameFactory, metadata: ProductMetadata) -> None:
        super().__init__('Metadata')  # FIXME snake_case?
        self._parent = parent
        self._name = self._registerStringParameter('Name', parent.createUniqueName(metadata.name))
        self.comments = self._registerStringParameter('Comments', metadata.comments)
        self.probeEnergyInElectronVolts = self._registerRealParameter(
            'ProbeEnergyInElectronVolts', metadata.probeEnergyInElectronVolts, minimum=0.)
        self.detectorDistanceInMeters = self._registerRealParameter(
            'DetectorDistanceInMeters', metadata.detectorDistanceInMeters, minimum=0.)
        self._index = -1

    def copy(self) -> MetadataRepositoryItem:
        return MetadataRepositoryItem(self._parent, self.getMetadata())

    def getName(self) -> str:
        return self._name.getValue()

    def setName(self, name: str) -> None:
        uniqueName = self._parent.createUniqueName(name)
        self._name.setValue(uniqueName)

    def getIndex(self) -> int:
        return self._index

    def getMetadata(self) -> ProductMetadata:
        return ProductMetadata(
            name=self._name.getValue(),
            comments=self.comments.getValue(),
            probeEnergyInElectronVolts=self.probeEnergyInElectronVolts.getValue(),
            detectorDistanceInMeters=self.detectorDistanceInMeters.getValue(),
        )

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
        super().__init__('metadata')
        self._parent = parent
        self._name = self._registerStringParameter('name', parent.createUniqueName(metadata.name))
        self.comments = self._registerStringParameter('comments', metadata.comments)
        self.detectorDistanceInMeters = self._registerRealParameter(
            'detector_distance_m', metadata.detectorDistanceInMeters, minimum=0.)
        self.probeEnergyInElectronVolts = self._registerRealParameter(
            'probe_energy_eV', metadata.probeEnergyInElectronVolts, minimum=0.)
        self.probePhotonsPerSecond = self._registerRealParameter('probe_photons_per_second',
                                                                 metadata.probePhotonsPerSecond,
                                                                 minimum=0.)
        self.exposureTimeInSeconds = self._registerRealParameter('exposure_time_s',
                                                                 metadata.exposureTimeInSeconds,
                                                                 minimum=0.)

        self._index = -1

    def assign(self, item: MetadataRepositoryItem) -> None:
        self.setName(item.getName())
        self.comments.setValue(item.comments.getValue())
        self.detectorDistanceInMeters.setValue(item.detectorDistanceInMeters.getValue())
        self.probeEnergyInElectronVolts.setValue(item.probeEnergyInElectronVolts.getValue())
        self.probePhotonsPerSecond.setValue(item.probePhotonsPerSecond.getValue())
        self.exposureTimeInSeconds.setValue(item.exposureTimeInSeconds.getValue())

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
            detectorDistanceInMeters=self.detectorDistanceInMeters.getValue(),
            probeEnergyInElectronVolts=self.probeEnergyInElectronVolts.getValue(),
            probePhotonsPerSecond=self.probePhotonsPerSecond.getValue(),
            exposureTimeInSeconds=self.exposureTimeInSeconds.getValue(),
        )

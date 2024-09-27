from __future__ import annotations
from abc import ABC, abstractmethod
import logging

from ptychodus.api.parametric import ParameterGroup, RealParameter, StringParameter
from ptychodus.api.product import ProductMetadata

logger = logging.getLogger(__name__)


class UniqueNameFactory(ABC):
    @abstractmethod
    def createUniqueName(self, candidateName: str) -> str:
        pass


class MetadataRepositoryItem(ParameterGroup):
    def __init__(self, parent: UniqueNameFactory, metadata: ProductMetadata) -> None:
        super().__init__()
        self._parent = parent
        self._name = StringParameter(self, 'name', parent.createUniqueName(metadata.name))
        self.comments = StringParameter(self, 'comments', metadata.comments)
        self.detectorDistanceInMeters = RealParameter(
            self, 'detector_distance_m', metadata.detectorDistanceInMeters, minimum=0.0
        )
        self.probeEnergyInElectronVolts = RealParameter(
            self, 'probe_energy_eV', metadata.probeEnergyInElectronVolts, minimum=0.0
        )
        self.probePhotonsPerSecond = RealParameter(
            self,
            'probe_photons_per_second',
            metadata.probePhotonsPerSecond,
            minimum=0.0,
        )
        self.exposureTimeInSeconds = RealParameter(
            self, 'exposure_time_s', metadata.exposureTimeInSeconds, minimum=0.0
        )

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

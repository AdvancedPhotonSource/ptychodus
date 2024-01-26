from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from .object import Object
from .probe import Probe
from .scan import Scan


@dataclass(frozen=True)
class ArtifactMetadata:
    name: str
    comments: str
    probeEnergyInElectronVolts: float
    detectorObjectDistanceInMeters: float


@dataclass(frozen=True)
class Artifact:
    metadata: ArtifactMetadata
    scan: Scan
    probe: Probe
    object_: Object


class ArtifactFileReader(ABC):

    @abstractmethod
    def read(self, filePath: Path) -> Artifact:
        '''reads an artifact from file'''
        pass


class ArtifactFileWriter(ABC):

    @abstractmethod
    def write(self, filePath: Path, artifact: Artifact) -> None:
        '''writes an artifact to file'''
        pass

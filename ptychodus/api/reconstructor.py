from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .data import DiffractionPatternArrayType
from .image import ImageExtent
from .object import ObjectArrayType, ObjectInterpolator
from .plot import Plot2D
from .probe import ProbeArrayType
from .scan import Scan


@dataclass(frozen=True)
class ReconstructInput:
    diffractionPatternArray: DiffractionPatternArrayType
    scan: Scan
    probeArray: ProbeArrayType
    objectInterpolator: ObjectInterpolator

    @property
    def diffractionPatternExtent(self) -> ImageExtent:
        return ImageExtent(
            width=self.diffractionPatternArray.shape[-1],
            height=self.diffractionPatternArray.shape[-2],
        )

    @property
    def probeExtent(self) -> ImageExtent:
        return ImageExtent(
            width=self.probeArray.shape[-1],
            height=self.probeArray.shape[-2],
        )

    @property
    def objectArray(self) -> ObjectArrayType:
        return self.objectInterpolator.getArray()

    @property
    def objectExtent(self) -> ImageExtent:
        return ImageExtent(
            width=self.objectArray.shape[-1],
            height=self.objectArray.shape[-2],
        )


@dataclass(frozen=True)
class ReconstructOutput:
    scan: Scan | None
    probeArray: ProbeArrayType | None
    objectArray: ObjectArrayType | None
    objective: Sequence[Sequence[float]]
    plot2D: Plot2D
    result: int

    @classmethod
    def createNull(cls) -> ReconstructOutput:
        return cls(None, None, None, [[]], Plot2D.createNull(), 0)


class Reconstructor(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def reconstruct(self, parameters: ReconstructInput) -> ReconstructOutput:
        pass


class TrainableReconstructor(Reconstructor):

    @abstractmethod
    def ingestTrainingData(self, parameters: ReconstructInput) -> None:
        pass

    @abstractmethod
    def getSaveFileFilterList(self) -> Sequence[str]:
        pass

    @abstractmethod
    def getSaveFileFilter(self) -> str:
        pass

    @abstractmethod
    def saveTrainingData(self, filePath: Path) -> None:
        pass

    @abstractmethod
    def train(self) -> Plot2D:
        pass

    @abstractmethod
    def clearTrainingData(self) -> None:
        pass


class NullReconstructor(TrainableReconstructor):

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def reconstruct(self, parameters: ReconstructInput) -> ReconstructOutput:
        return ReconstructOutput(
            scan=parameters.scan,
            probeArray=parameters.probeArray,
            objectArray=parameters.objectInterpolator.getArray(),
            objective=[[]],
            plot2D=Plot2D.createNull(),
            result=0,
        )

    def ingestTrainingData(self, parameters: ReconstructInput) -> None:
        pass

    def getSaveFileFilterList(self) -> Sequence[str]:
        return list()

    def getSaveFileFilter(self) -> str:
        return str()

    def saveTrainingData(self, filePath: Path) -> None:
        pass

    def train(self) -> Plot2D:
        return Plot2D.createNull()

    def clearTrainingData(self) -> None:
        pass


class ReconstructorLibrary(Iterable[Reconstructor]):

    @property
    @abstractmethod
    def name(self) -> str:
        pass

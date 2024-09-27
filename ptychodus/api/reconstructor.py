from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .product import Product
from .patterns import DiffractionPatternArrayType


@dataclass(frozen=True)
class ReconstructInput:
    patterns: DiffractionPatternArrayType
    product: Product


@dataclass(frozen=True)
class ReconstructOutput:
    product: Product
    result: int


class Reconstructor(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def reconstruct(self, parameters: ReconstructInput) -> ReconstructOutput:
        pass


@dataclass(frozen=True)
class TrainOutput:
    trainingLoss: Sequence[float]
    validationLoss: Sequence[float]
    result: int


class TrainableReconstructor(Reconstructor):
    @abstractmethod
    def ingestTrainingData(self, parameters: ReconstructInput) -> None:
        pass

    @abstractmethod
    def getOpenTrainingDataFileFilterList(self) -> Sequence[str]:
        pass

    @abstractmethod
    def getOpenTrainingDataFileFilter(self) -> str:
        pass

    @abstractmethod
    def openTrainingData(self, filePath: Path) -> None:
        pass

    @abstractmethod
    def getSaveTrainingDataFileFilterList(self) -> Sequence[str]:
        pass

    @abstractmethod
    def getSaveTrainingDataFileFilter(self) -> str:
        pass

    @abstractmethod
    def saveTrainingData(self, filePath: Path) -> None:
        pass

    @abstractmethod
    def train(self) -> TrainOutput:
        pass

    @abstractmethod
    def clearTrainingData(self) -> None:
        pass

    @abstractmethod
    def getOpenModelFileFilterList(self) -> Sequence[str]:
        pass

    @abstractmethod
    def getOpenModelFileFilter(self) -> str:
        pass

    @abstractmethod
    def openModel(self, filePath: Path) -> None:
        pass

    @abstractmethod
    def getSaveModelFileFilterList(self) -> Sequence[str]:
        pass

    @abstractmethod
    def getSaveModelFileFilter(self) -> str:
        pass

    @abstractmethod
    def saveModel(self, filePath: Path) -> None:
        pass


class NullReconstructor(TrainableReconstructor):
    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def reconstruct(self, parameters: ReconstructInput) -> ReconstructOutput:
        return ReconstructOutput(parameters.product, 0)

    def ingestTrainingData(self, parameters: ReconstructInput) -> None:
        pass

    def getOpenTrainingDataFileFilterList(self) -> Sequence[str]:
        return list()

    def getOpenTrainingDataFileFilter(self) -> str:
        return str()

    def openTrainingData(self, filePath: Path) -> None:
        pass

    def getSaveTrainingDataFileFilterList(self) -> Sequence[str]:
        return list()

    def getSaveTrainingDataFileFilter(self) -> str:
        return str()

    def saveTrainingData(self, filePath: Path) -> None:
        pass

    def train(self) -> TrainOutput:
        return TrainOutput([], [], 0)

    def clearTrainingData(self) -> None:
        pass

    def getOpenModelFileFilterList(self) -> Sequence[str]:
        return list()

    def getOpenModelFileFilter(self) -> str:
        return str()

    def openModel(self, filePath: Path) -> None:
        pass

    def getSaveModelFileFilterList(self) -> Sequence[str]:
        return list()

    def getSaveModelFileFilter(self) -> str:
        return str()

    def saveModel(self, filePath: Path) -> None:
        pass


class ReconstructorLibrary(Iterable[Reconstructor]):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

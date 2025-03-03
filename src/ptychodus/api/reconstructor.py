from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .product import Product
from .patterns import BooleanArrayType, PatternDataType


@dataclass(frozen=True)
class ReconstructInput:
    patterns: PatternDataType
    bad_pixels: BooleanArrayType
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
    def getModelFileFilter(self) -> str:
        pass

    @abstractmethod
    def openModel(self, filePath: Path) -> None:
        pass

    @abstractmethod
    def saveModel(self, filePath: Path) -> None:
        pass

    @abstractmethod
    def getTrainingDataFileFilter(self) -> str:
        pass

    @abstractmethod
    def exportTrainingData(self, filePath: Path, parameters: ReconstructInput) -> None:
        pass

    @abstractmethod
    def getTrainingDataPath(self) -> Path:
        pass

    @abstractmethod
    def train(self, dataPath: Path) -> TrainOutput:
        pass


class NullReconstructor(TrainableReconstructor):
    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def reconstruct(self, parameters: ReconstructInput) -> ReconstructOutput:
        return ReconstructOutput(parameters.product, 0)

    def getModelFileFilter(self) -> str:
        return str()

    def openModel(self, filePath: Path) -> None:
        pass

    def saveModel(self, filePath: Path) -> None:
        pass

    def getTrainingDataFileFilter(self) -> str:
        return str()

    def exportTrainingData(self, filePath: Path, parameters: ReconstructInput) -> None:
        pass

    def getTrainingDataPath(self) -> Path:
        return Path()

    def train(self, dataPath: Path) -> TrainOutput:
        return TrainOutput([], [], 0)


class ReconstructorLibrary(Iterable[Reconstructor]):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def logger_name(self) -> str:
        pass

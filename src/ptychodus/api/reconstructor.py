from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from ptychodus.api.typing import BooleanArrayType

from .patterns import PatternDataType
from .product import Product


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
class LossValue:
    epoch: int
    training_loss: float
    validation_loss: float


@dataclass(frozen=True)
class TrainOutput:
    losses: Sequence[LossValue]
    result: int


class TrainableReconstructor(Reconstructor):
    @abstractmethod
    def get_model_file_filter(self) -> str:
        pass

    @abstractmethod
    def open_model(self, file_path: Path) -> None:
        pass

    @abstractmethod
    def save_model(self, file_path: Path) -> None:
        pass

    @abstractmethod
    def get_training_data_file_filter(self) -> str:
        pass

    @abstractmethod
    def export_training_data(self, file_path: Path, parameters: ReconstructInput) -> None:
        pass

    @abstractmethod
    def get_training_data_path(self) -> Path:
        pass

    @abstractmethod
    def train(self, data_path: Path) -> TrainOutput:
        pass


class NullReconstructor(TrainableReconstructor):
    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def reconstruct(self, parameters: ReconstructInput) -> ReconstructOutput:
        return ReconstructOutput(parameters.product, 0)

    def get_model_file_filter(self) -> str:
        return str()

    def open_model(self, file_path: Path) -> None:
        pass

    def save_model(self, file_path: Path) -> None:
        pass

    def get_training_data_file_filter(self) -> str:
        return str()

    def export_training_data(self, file_path: Path, parameters: ReconstructInput) -> None:
        pass

    def get_training_data_path(self) -> Path:
        return Path()

    def train(self, data_path: Path) -> TrainOutput:
        return TrainOutput([], 0)


class ReconstructorLibrary(Iterable[Reconstructor]):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def logger_name(self) -> str:
        pass

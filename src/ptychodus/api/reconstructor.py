from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
import logging

from .diffraction import BadPixels, DiffractionPatterns
from .product import LossValue, Product


@dataclass(frozen=True)
class ReconstructInput:
    diffraction_patterns: DiffractionPatterns
    bad_pixels: BadPixels
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
    training_loss: Sequence[LossValue]
    validation_loss: Sequence[LossValue]
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
        return TrainOutput([], [], 0)


class ReconstructorLibrary(Iterable[Reconstructor]):
    def __init__(self, logger_name: str) -> None:
        self._logger = logging.getLogger(logger_name)

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    def log_levels(self) -> Iterable[str]:
        return ('CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG')

    def get_logger(self) -> logging.Logger:
        return self._logger

    def get_log_level(self) -> str:
        level = self._logger.getEffectiveLevel()
        return logging.getLevelName(level)

    def set_log_level(self, name: str) -> None:
        name_before = self.get_log_level()

        try:
            self._logger.setLevel(name)
        except ValueError:
            self._logger.error(f'Bad log level "{name}".')

        name_after = self.get_log_level()
        self._logger.info(f'Changed {self.name} logging level {name_before} -> {name_after}')

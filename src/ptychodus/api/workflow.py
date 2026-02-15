from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from ptychodus.api.diffraction import CropCenter
from ptychodus.api.geometry import AffineTransform, ImageExtent
from ptychodus.api.product import Product
from ptychodus.api.reconstructor import AssembledDiffractionData, ReconstructInput
from ptychodus.api.settings import PathPrefixChange


class WorkflowDiffractionAPI(ABC):
    @abstractmethod
    def get_assembled_data(self) -> AssembledDiffractionData:
        pass

    @abstractmethod
    def save_assembled_data(self, file_path: Path) -> None:
        pass


class WorkflowProductAPI(ABC):
    @abstractmethod
    def get_product_index(self) -> int:
        pass

    @abstractmethod
    def get_product(self) -> Product:
        pass

    @abstractmethod
    def rename_product(self, new_name: str) -> None:
        pass

    @abstractmethod
    def load_probe_positions(self, file_path: Path, *, file_type: str | None = None) -> None:
        pass

    @abstractmethod
    def generate_probe_positions(
        self, generator_name: str | None = None, generator_parameters: Mapping[str, Any] = {}
    ) -> None:
        pass

    @abstractmethod
    def set_probe_positions_transform(self, transform: AffineTransform) -> None:
        pass

    @abstractmethod
    def load_probe(self, file_path: Path, *, file_type: str | None = None) -> None:
        pass

    @abstractmethod
    def generate_probe(
        self, generator_name: str | None = None, generator_parameters: Mapping[str, Any] = {}
    ) -> None:
        pass

    @abstractmethod
    def load_object(self, file_path: Path, *, file_type: str | None = None) -> None:
        pass

    @abstractmethod
    def generate_object(
        self, generator_name: str | None = None, generator_parameters: Mapping[str, Any] = {}
    ) -> None:
        pass

    @abstractmethod
    def get_reconstruct_input(self) -> ReconstructInput:
        pass

    @abstractmethod
    def reconstruct_local(
        self,
        *,
        algorithm: str | None = None,
        output_product_file: Path | None = None,
        block: bool = False,
    ) -> WorkflowProductAPI:
        pass

    @abstractmethod
    def reconstruct_remote(self, *, algorithm: str | None = None) -> None:
        pass

    @abstractmethod
    def train_reconstructor_local(
        self,
        input_path: Path,
        output_path: Path,
        *,
        algorithm: str | None = None,
        block: bool = False,
    ) -> None:
        pass

    @abstractmethod
    def train_reconstructor_remote(self, *, algorithm: str | None = None) -> None:
        pass

    @abstractmethod
    def export_training_data(self, file_path: Path, *, algorithm: str | None = None) -> None:
        pass

    @abstractmethod
    def save_product(self, file_path: Path, *, file_type: str | None = None) -> None:
        pass


class WorkflowAPI(ABC):
    @abstractmethod
    def load_diffraction_data(
        self,
        file_path: Path,
        *,
        file_type: str | None = None,
        crop_center: CropCenter | None = None,
        crop_extent: ImageExtent | None = None,
        detector_extent: ImageExtent | None = None,
        process_patterns: bool = True,
        block: bool = False,
    ) -> WorkflowDiffractionAPI:
        pass

    @abstractmethod
    def load_assembled_diffraction_data(self, file_path: Path) -> WorkflowDiffractionAPI:
        pass

    @abstractmethod
    def register_product(self, product: Product) -> WorkflowProductAPI:
        pass

    @abstractmethod
    def load_product(self, file_path: Path, *, file_type: str | None = None) -> WorkflowProductAPI:
        pass

    @abstractmethod
    def create_product(
        self,
        name: str,
        *,
        comments: str = '',
        detector_distance_m: float | None = None,
        probe_energy_eV: float | None = None,  # noqa: N803
        probe_photon_count: float | None = None,
        exposure_time_s: float | None = None,
        mass_attenuation_m2_kg: float | None = None,
        tomography_angle_deg: float | None = None,
    ) -> WorkflowProductAPI:
        pass

    @abstractmethod
    def get_product(self, product_index: int) -> WorkflowProductAPI:
        pass

    @abstractmethod
    def available_reconstructors(self) -> Iterator[str]:
        pass

    @abstractmethod
    def save_settings(
        self, file_path: Path, change_path_prefix: PathPrefixChange | None = None
    ) -> None:
        pass


class FileBasedWorkflow(ABC):
    @property
    @abstractmethod
    def is_watch_recursive(self) -> bool:
        """indicates whether the data directory must be watched recursively"""
        pass

    @abstractmethod
    def get_watch_file_pattern(self) -> str:
        """UNIX-style filename pattern. For rules see fnmatch from Python standard library."""
        pass

    @abstractmethod
    def execute(self, api: WorkflowAPI, file_path: Path) -> None:
        """uses workflow API to execute the workflow"""
        pass

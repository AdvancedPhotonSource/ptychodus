"""Workflow API interfaces for orchestrating ptychography data processing pipelines."""

from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from pathlib import Path
from enum import Enum, auto
from typing import Any

from ptychodus.api.diffraction import CropCenter
from ptychodus.api.geometry import AffineTransform, ImageExtent
from ptychodus.api.product import Product
from ptychodus.api.reconstructor import AssembledDiffractionData, ReconstructInput
from ptychodus.api.settings import PathPrefixChange


class RemoteComputeProvider(Enum):
    GLOBUS = auto()
    GENESIS = auto()


class WorkflowDiffractionAPI(ABC):
    """Act on an assembled diffraction dataset within a workflow."""

    @abstractmethod
    def get_assembled_data(self) -> AssembledDiffractionData:
        """Return the assembled diffraction data."""
        pass

    @abstractmethod
    def save_assembled_data(self, file_path: Path) -> None:
        """Save the assembled diffraction data to file."""
        pass


class WorkflowProductAPI(ABC):
    """Act on a data product within a workflow."""

    @abstractmethod
    def get_product_index(self) -> int:
        """Return the registry index of this product."""
        pass

    @abstractmethod
    def get_product(self) -> Product:
        """Return the current product."""
        pass

    @abstractmethod
    def rename_product(self, new_name: str) -> None:
        """Rename the product."""
        pass

    @abstractmethod
    def load_probe_positions(self, file_path: Path, *, file_type: str | None = None) -> None:
        """Load probe positions from file, uses format from settings when file_type is None."""
        pass

    @abstractmethod
    def generate_probe_positions(
        self,
        generator_name: str | None = None,
        generator_parameters: Mapping[str, Any] | None = None,
    ) -> None:
        """Generate probe positions using the named generator and optional parameter overrides."""
        pass

    @abstractmethod
    def set_probe_positions_transform(self, transform: AffineTransform) -> None:
        """Apply an affine transform to the probe positions."""
        pass

    @abstractmethod
    def load_probe(self, file_path: Path, *, file_type: str | None = None) -> None:
        """Load a probe sequence from file, uses format from settings when file_type is None."""
        pass

    @abstractmethod
    def generate_probe(
        self,
        generator_name: str | None = None,
        generator_parameters: Mapping[str, Any] | None = None,
    ) -> None:
        """Generate a probe using the named generator and optional parameter overrides."""
        pass

    @abstractmethod
    def load_object(self, file_path: Path, *, file_type: str | None = None) -> None:
        """Load an object array from file, uses format from settings when file_type is None."""
        pass

    @abstractmethod
    def generate_object(
        self,
        generator_name: str | None = None,
        generator_parameters: Mapping[str, Any] | None = None,
    ) -> None:
        """Generate an object using the named generator and optional parameter overrides."""
        pass

    @abstractmethod
    def get_reconstruct_input(self) -> ReconstructInput:
        """Assemble and return the reconstruction input for this product."""
        pass

    @abstractmethod
    def reconstruct_local(
        self,
        *,
        algorithm: str | None = None,
        output_product_file: Path | None = None,
        block: bool = False,
    ) -> WorkflowProductAPI:
        """Run reconstruction locally; returns a handle to the output product.

        Blocks until completion when block is True, otherwise returns immediately.
        """
        pass

    @abstractmethod
    def reconstruct_remote(
        self,
        *,
        algorithm: str | None = None,
        provider: RemoteComputeProvider = RemoteComputeProvider.GLOBUS,
    ) -> None:
        """Submit reconstruction to a remote compute resource."""
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
        """Train the reconstructor model locally from input_path and write to output_path.

        Blocks until completion when block is True, otherwise returns immediately.
        """
        pass

    @abstractmethod
    def train_reconstructor_remote(
        self,
        *,
        algorithm: str | None = None,
        provider: RemoteComputeProvider = RemoteComputeProvider.GLOBUS,
    ) -> None:
        """Submit reconstructor training to a remote compute resource."""
        pass

    @abstractmethod
    def export_training_data(self, file_path: Path, *, algorithm: str | None = None) -> None:
        """Export training data for the current reconstructor to file."""
        pass

    @abstractmethod
    def save_product(self, file_path: Path, *, file_type: str | None = None) -> None:
        """Save the product to file, uses format from settings when file_type is None."""
        pass


class WorkflowAPI(ABC):
    """Top-level API for loading data, managing products, and running reconstructions."""

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
        """Load and assemble a diffraction dataset from file.

        Blocks until complete when block is True, otherwise returns immediately.
        """
        pass

    @abstractmethod
    def load_assembled_diffraction_data(self, file_path: Path) -> WorkflowDiffractionAPI:
        """Load a pre-assembled diffraction dataset from file."""
        pass

    @abstractmethod
    def register_product(self, product: Product) -> WorkflowProductAPI:
        """Register an existing Product object and return a handle to it."""
        pass

    @abstractmethod
    def load_product(self, file_path: Path, *, file_type: str | None = None) -> WorkflowProductAPI:
        """Load a product from file and return a handle to it."""
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
        """Create a new product with optional metadata overrides and return a handle to it."""
        pass

    @abstractmethod
    def get_product(self, product_index: int) -> WorkflowProductAPI:
        """Return a handle to an already-registered product by index."""
        pass

    @abstractmethod
    def available_reconstructors(self) -> Iterator[str]:
        """Yield the names of all registered reconstructors."""
        pass

    @abstractmethod
    def save_settings(
        self, file_path: Path, change_path_prefix: PathPrefixChange | None = None
    ) -> None:
        """Save the current settings to file, optionally remapping path prefixes."""
        pass


class FileBasedWorkflow(ABC):
    """Plugin interface for file-based workflows."""

    @property
    @abstractmethod
    def is_watch_recursive(self) -> bool:
        """True if the data directory must be watched recursively."""
        pass

    @abstractmethod
    def get_watch_file_pattern(self) -> str:
        """UNIX-style filename pattern. For rules see fnmatch from Python standard library."""
        pass

    @abstractmethod
    def execute(self, api: WorkflowAPI, file_path: Path) -> None:
        """Execute the workflow using the provided WorkflowAPI."""
        pass

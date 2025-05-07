from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ptychodus.api.geometry import ImageExtent
from ptychodus.api.patterns import CropCenter
from ptychodus.api.settings import PathPrefixChange


class WorkflowProductAPI(ABC):
    @abstractmethod
    def open_scan(self, file_path: Path, *, file_type: str | None = None) -> None:
        pass

    @abstractmethod
    def build_scan(
        self, builder_name: str | None = None, builder_parameters: Mapping[str, Any] = {}
    ) -> None:
        pass

    @abstractmethod
    def open_probe(self, file_path: Path, *, file_type: str | None = None) -> None:
        pass

    @abstractmethod
    def build_probe(
        self, builder_name: str | None = None, builder_parameters: Mapping[str, Any] = {}
    ) -> None:
        pass

    @abstractmethod
    def open_object(self, file_path: Path, *, file_type: str | None = None) -> None:
        pass

    @abstractmethod
    def build_object(
        self, builder_name: str | None = None, builder_parameters: Mapping[str, Any] = {}
    ) -> None:
        pass

    @abstractmethod
    def reconstruct_local(self) -> WorkflowProductAPI:
        pass

    @abstractmethod
    def reconstruct_remote(self) -> None:
        pass

    @abstractmethod
    def save_product(self, file_path: Path, *, file_type: str | None = None) -> None:
        pass


class WorkflowAPI(ABC):
    @abstractmethod
    def open_patterns(
        self,
        file_path: Path,
        *,
        file_type: str | None = None,
        crop_center: CropCenter | None = None,
        crop_extent: ImageExtent | None = None,
    ) -> None:
        """opens diffraction patterns from file"""
        pass

    @abstractmethod
    def import_assembled_patterns(self, file_path: Path) -> None:
        """import assembled patterns"""
        pass

    @abstractmethod
    def export_assembled_patterns(self, file_path: Path) -> None:
        """export assembled patterns"""
        pass

    @abstractmethod
    def open_product(self, file_path: Path, *, file_type: str | None = None) -> WorkflowProductAPI:
        """opens product from file"""
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
        """creates a new product"""
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

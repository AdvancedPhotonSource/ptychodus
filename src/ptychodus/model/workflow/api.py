from __future__ import annotations
from collections.abc import Mapping
from pathlib import Path
from typing import Any
import logging

from ptychodus.api.geometry import ImageExtent
from ptychodus.api.patterns import CropCenter
from ptychodus.api.settings import PathPrefixChange, SettingsRegistry
from ptychodus.api.workflow import WorkflowAPI, WorkflowProductAPI

from ..patterns import PatternsAPI
from ..product import ObjectAPI, ProbeAPI, ProductAPI, ScanAPI
from ..reconstructor import ReconstructorAPI
from .executor import WorkflowExecutor

logger = logging.getLogger(__name__)


class ConcreteWorkflowProductAPI(WorkflowProductAPI):
    def __init__(
        self,
        product_api: ProductAPI,
        scan_api: ScanAPI,
        probe_api: ProbeAPI,
        object_api: ObjectAPI,
        reconstructor_api: ReconstructorAPI,
        executor: WorkflowExecutor,
        product_index: int,
    ) -> None:
        self._product_api = product_api
        self._scan_api = scan_api
        self._probe_api = probe_api
        self._object_api = object_api
        self._reconstructor_api = reconstructor_api
        self._executor = executor
        self._product_index = product_index

    def open_scan(self, file_path: Path, *, file_type: str | None = None) -> None:
        self._scan_api.open_scan(self._product_index, file_path, file_type=file_type)

    def build_scan(
        self, builder_name: str | None = None, builder_parameters: Mapping[str, Any] = {}
    ) -> None:
        if builder_name is None:
            self._scan_api.build_scan_from_settings(self._product_index)
        else:
            self._scan_api.build_scan(self._product_index, builder_name, builder_parameters)

    def open_probe(self, file_path: Path, *, file_type: str | None = None) -> None:
        self._probe_api.open_probe(self._product_index, file_path, file_type=file_type)

    def build_probe(
        self, builder_name: str | None = None, builder_parameters: Mapping[str, Any] = {}
    ) -> None:
        if builder_name is None:
            self._probe_api.build_probe_from_settings(self._product_index)
        else:
            self._probe_api.build_probe(self._product_index, builder_name, builder_parameters)

    def open_object(self, file_path: Path, *, file_type: str | None = None) -> None:
        self._object_api.open_object(self._product_index, file_path, file_type=file_type)

    def build_object(
        self, builder_name: str | None = None, builder_parameters: Mapping[str, Any] = {}
    ) -> None:
        if builder_name is None:
            self._object_api.build_object_from_settings(self._product_index)
        else:
            self._object_api.build_object(self._product_index, builder_name, builder_parameters)

    def reconstruct_local(self) -> WorkflowProductAPI:
        logger.debug(f'Reconstruct: index={self._product_index}')
        output_product_index = self._reconstructor_api.reconstruct(self._product_index)

        return ConcreteWorkflowProductAPI(
            self._product_api,
            self._scan_api,
            self._probe_api,
            self._object_api,
            self._reconstructor_api,
            self._executor,
            output_product_index,
        )

    def reconstruct_remote(self) -> None:
        logger.debug(f'Execute Workflow: index={self._product_index}')
        self._executor.run_flow(self._product_index)

    def save_product(self, file_path: Path, *, file_type: str | None = None) -> None:
        self._product_api.save_product(self._product_index, file_path, file_type=file_type)


class ConcreteWorkflowAPI(WorkflowAPI):
    def __init__(
        self,
        settings_registry: SettingsRegistry,
        patterns_api: PatternsAPI,
        product_api: ProductAPI,
        scan_api: ScanAPI,
        probe_api: ProbeAPI,
        object_api: ObjectAPI,
        reconstructor_api: ReconstructorAPI,
        executor: WorkflowExecutor,
    ) -> None:
        self._settings_registry = settings_registry
        self._patterns_api = patterns_api
        self._product_api = product_api
        self._scan_api = scan_api
        self._probe_api = probe_api
        self._object_api = object_api
        self._reconstructor_api = reconstructor_api
        self._executor = executor

    def _create_product_api(self, product_index: int) -> WorkflowProductAPI:
        if product_index < 0:
            raise ValueError(f'Bad product index ({product_index=})!')

        return ConcreteWorkflowProductAPI(
            self._product_api,
            self._scan_api,
            self._probe_api,
            self._object_api,
            self._reconstructor_api,
            self._executor,
            product_index,
        )

    def open_patterns(
        self,
        file_path: Path,
        *,
        file_type: str | None = None,
        crop_center: CropCenter | None = None,
        crop_extent: ImageExtent | None = None,
    ) -> None:
        self._patterns_api.open_patterns(
            file_path, file_type=file_type, crop_center=crop_center, crop_extent=crop_extent
        )

    def import_assembled_patterns(self, file_path: Path) -> None:
        self._patterns_api.import_assembled_patterns(file_path)

    def export_assembled_patterns(self, file_path: Path) -> None:
        self._patterns_api.export_assembled_patterns(file_path)

    def open_product(self, file_path: Path, *, file_type: str | None = None) -> WorkflowProductAPI:
        product_index = self._product_api.open_product(file_path, file_type=file_type)
        return self._create_product_api(product_index)

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
        product_index = self._product_api.insert_new_product(
            name,
            comments=comments,
            detector_distance_m=detector_distance_m,
            probe_energy_eV=probe_energy_eV,
            probe_photon_count=probe_photon_count,
            exposure_time_s=exposure_time_s,
            mass_attenuation_m2_kg=mass_attenuation_m2_kg,
            tomography_angle_deg=tomography_angle_deg,
        )
        return self._create_product_api(product_index)

    def save_settings(
        self, file_path: Path, change_path_prefix: PathPrefixChange | None = None
    ) -> None:
        self._settings_registry.save_settings(file_path, change_path_prefix)

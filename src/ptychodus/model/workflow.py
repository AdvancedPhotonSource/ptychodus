from __future__ import annotations
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any
import logging

from ptychodus.api.diffraction import CropCenter
from ptychodus.api.geometry import AffineTransform, ImageExtent
from ptychodus.api.product import Product
from ptychodus.api.reconstructor import AssembledDiffractionData, ReconstructInput
from ptychodus.api.settings import PathPrefixChange, SettingsRegistry
from ptychodus.api.workflow import (
    RemoteComputeProvider,
    WorkflowAPI,
    WorkflowDiffractionAPI,
    WorkflowProductAPI,
)

from .diffraction import DiffractionAPI
from .genesis import GenesisExecutor
from .globus import GlobusExecutor
from .processing import ProcessingAPI
from .product import ObjectAPI, ProbeAPI, ProductAPI, ProbePositionsAPI

logger = logging.getLogger(__name__)


class ConcreteWorkflowDiffractionAPI(WorkflowDiffractionAPI):
    def __init__(self, diffraction_api: DiffractionAPI) -> None:
        self._diffraction_api = diffraction_api

    def get_assembled_data(self) -> AssembledDiffractionData:
        return self._diffraction_api.get_assembled_data()

    def save_assembled_data(self, file_path: Path) -> None:
        self._diffraction_api.export_assembled_patterns(file_path)


class ConcreteWorkflowProductAPI(WorkflowProductAPI):
    def __init__(
        self,
        product_api: ProductAPI,
        probe_positions_api: ProbePositionsAPI,
        probe_api: ProbeAPI,
        object_api: ObjectAPI,
        processing_api: ProcessingAPI,
        globus_executor: GlobusExecutor,
        genesis_executor: GenesisExecutor,
        product_index: int,
    ) -> None:
        self._product_api = product_api
        self._probe_positions_api = probe_positions_api
        self._probe_api = probe_api
        self._object_api = object_api
        self._processing_api = processing_api
        self._globus_executor = globus_executor
        self._genesis_executor = genesis_executor
        self._product_index = product_index

    def get_product_index(self) -> int:
        return self._product_index

    def get_product(self) -> Product:
        item = self._product_api.get_item(self._product_index)
        return item.get_product()

    def rename_product(self, new_name: str) -> None:
        self._product_api.rename_product(self._product_index, new_name)

    def load_probe_positions(self, file_path: Path, *, file_type: str | None = None) -> None:
        self._probe_positions_api.open_probe_positions(
            self._product_index, file_path, file_type=file_type
        )

    def generate_probe_positions(
        self,
        generator_name: str | None = None,
        generator_parameters: Mapping[str, Any] | None = None,
    ) -> None:
        if generator_name is None:
            self._probe_positions_api.build_probe_positions_from_settings(self._product_index)
        else:
            self._probe_positions_api.build_probe_positions(
                self._product_index, generator_name, generator_parameters
            )

    def set_probe_positions_transform(self, transform: AffineTransform) -> None:
        item = self._product_api.get_item(self._product_index)
        builder = item.get_probe_positions_item().get_builder()
        builder.set_transform(transform)

    def load_probe(self, file_path: Path, *, file_type: str | None = None) -> None:
        self._probe_api.open_probe(self._product_index, file_path, file_type=file_type)

    def generate_probe(
        self,
        generator_name: str | None = None,
        generator_parameters: Mapping[str, Any] | None = None,
    ) -> None:
        if generator_name is None:
            self._probe_api.build_probe_from_settings(self._product_index)
        else:
            self._probe_api.build_probe(self._product_index, generator_name, generator_parameters)

    def load_object(self, file_path: Path, *, file_type: str | None = None) -> None:
        self._object_api.open_object(self._product_index, file_path, file_type=file_type)

    def generate_object(
        self,
        generator_name: str | None = None,
        generator_parameters: Mapping[str, Any] | None = None,
    ) -> None:
        if generator_name is None:
            self._object_api.build_object_from_settings(self._product_index)
        else:
            self._object_api.build_object(self._product_index, generator_name, generator_parameters)

    def get_reconstruct_input(self) -> ReconstructInput:
        return self._processing_api.get_reconstruct_input(self._product_index)

    def reconstruct_local(
        self,
        *,
        algorithm: str | None = None,
        output_product_file: Path | None = None,
        block: bool = False,
    ) -> WorkflowProductAPI:
        output_product_index = self._processing_api.reconstruct(
            self._product_index,
            algorithm=algorithm,
            output_product_file=output_product_file,
            block=block,
        )

        return ConcreteWorkflowProductAPI(
            self._product_api,
            self._probe_positions_api,
            self._probe_api,
            self._object_api,
            self._processing_api,
            self._globus_executor,
            self._genesis_executor,
            output_product_index,
        )

    def reconstruct_remote(
        self,
        *,
        algorithm: str | None = None,
        provider: RemoteComputeProvider = RemoteComputeProvider.GLOBUS,
    ) -> None:
        match provider:
            case RemoteComputeProvider.GLOBUS:
                self._globus_executor.reconstruct(self._product_index, algorithm=algorithm)
            case RemoteComputeProvider.GENESIS:
                self._genesis_executor.reconstruct(self._product_index, algorithm=algorithm)
            case _:
                raise ValueError(f'Unsupported remote compute provider: {provider}')

    def train_reconstructor_local(
        self,
        input_path: Path,
        output_path: Path,
        *,
        algorithm: str | None = None,
        block: bool = False,
    ) -> None:
        # TODO mlflow
        self._processing_api.train(
            self._product_index, input_path, output_path, algorithm=algorithm, block=block
        )

    def train_reconstructor_remote(
        self,
        *,
        algorithm: str | None = None,
        provider: RemoteComputeProvider = RemoteComputeProvider.GLOBUS,
    ) -> None:
        # TODO mlflow
        match provider:
            case RemoteComputeProvider.GLOBUS:
                self._globus_executor.train(self._product_index, algorithm=algorithm)
            case RemoteComputeProvider.GENESIS:
                self._genesis_executor.train(self._product_index, algorithm=algorithm)
            case _:
                raise ValueError(f'Unsupported remote compute provider: {provider}')

    def export_training_data(self, file_path: Path, *, algorithm: str | None = None) -> None:
        self._processing_api.export_training_data(
            file_path, self._product_index, algorithm=algorithm
        )

    def save_product(self, file_path: Path, *, file_type: str | None = None) -> None:
        self._product_api.save_product(self._product_index, file_path, file_type=file_type)


class ConcreteWorkflowAPI(WorkflowAPI):
    def __init__(
        self,
        settings_registry: SettingsRegistry,
        diffraction_api: DiffractionAPI,
        product_api: ProductAPI,
        probe_positions_api: ProbePositionsAPI,
        probe_api: ProbeAPI,
        object_api: ObjectAPI,
        processing_api: ProcessingAPI,
        globus_executor: GlobusExecutor,
        genesis_executor: GenesisExecutor,
    ) -> None:
        self._settings_registry = settings_registry
        self._diffraction_api = diffraction_api
        self._product_api = product_api
        self._probe_positions_api = probe_positions_api
        self._probe_api = probe_api
        self._object_api = object_api
        self._processing_api = processing_api
        self._globus_executor = globus_executor
        self._genesis_executor = genesis_executor

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
        self._diffraction_api.open_patterns(
            file_path,
            file_type=file_type,
            crop_center=crop_center,
            crop_extent=crop_extent,
            detector_extent=detector_extent,
            process_patterns=process_patterns,
            block=block,
        )
        return ConcreteWorkflowDiffractionAPI(self._diffraction_api)

    def available_reconstructors(self) -> Iterator[str]:
        return self._processing_api.available_reconstructors()

    def load_assembled_diffraction_data(self, file_path: Path) -> WorkflowDiffractionAPI:
        self._diffraction_api.import_assembled_patterns(file_path)
        return ConcreteWorkflowDiffractionAPI(self._diffraction_api)

    def get_product(self, product_index: int) -> WorkflowProductAPI:
        if product_index < 0:
            raise ValueError(f'Bad product index ({product_index=})!')

        return ConcreteWorkflowProductAPI(
            self._product_api,
            self._probe_positions_api,
            self._probe_api,
            self._object_api,
            self._processing_api,
            self._globus_executor,
            self._genesis_executor,
            product_index,
        )

    def register_product(self, product: Product) -> WorkflowProductAPI:
        product_index = self._product_api.insert_product(product)
        return self.get_product(product_index)

    def load_product(self, file_path: Path, *, file_type: str | None = None) -> WorkflowProductAPI:
        product_index = self._product_api.open_product(file_path, file_type=file_type)

        if product_index < 0:
            raise RuntimeError(f'Failed to open product "{file_path}"!')

        return self.get_product(product_index)

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
        return self.get_product(product_index)

    def save_settings(
        self, file_path: Path, change_path_prefix: PathPrefixChange | None = None
    ) -> None:
        self._settings_registry.save_settings(file_path, change_path_prefix)

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Final
from uuid import UUID
import logging

from ptychodus.api.io import StandardFileLayout

from .iri import IRIClient, JobAttributes, JobSpecification, ResourceSpecification, ResourceType
from .settings import GenesisSettings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GlobusCollection:
    id: UUID
    globus_path: str
    posix_path: Path


class IRIFacilityAdapter(ABC):
    def __init__(self, iri_client: IRIClient) -> None:
        self._iri_client = iri_client
        self._compute_resource_name_to_id_cache: dict[str, str] = {}
        self._compute_resource_id_to_name_cache: dict[str, str] = {}
        self._project_name_to_id_cache: dict[str, str] = {}
        self._project_id_to_name_cache: dict[str, str] = {}

    def get_iri_client(self) -> IRIClient:
        return self._iri_client

    def refresh_compute_resources(self) -> None:
        self._compute_resource_name_to_id_cache.clear()
        self._compute_resource_id_to_name_cache.clear()

        try:
            resources = self._iri_client.status.get_resources(resource_type=ResourceType.COMPUTE)
        except Exception:
            logger.warning('Failed to fetch dynamic compute resources', exc_info=True)
        else:
            for resource in resources:
                name_segments: list[str] = []

                if resource.group is not None and resource.group != 'computes':
                    name_segments.append(resource.group.title())

                if resource.name is None:
                    name_segments.append(resource.id)
                else:
                    name_segments.append(resource.name.title())

                name = ' '.join(name_segments)
                self._compute_resource_name_to_id_cache[name] = resource.id
                self._compute_resource_id_to_name_cache[resource.id] = name

        api_base_url = self._iri_client.get_api_base_url()
        logger.info(
            f'Fetched compute resources from {api_base_url}:\n'
            + json.dumps(self._compute_resource_name_to_id_cache, indent=2)
        )

    def compute_resource_names(self) -> Sequence[str]:
        if not self._compute_resource_name_to_id_cache:
            self.refresh_compute_resources()

        return sorted(self._compute_resource_name_to_id_cache.keys())

    def map_compute_resource_name_to_id(self, resource_name: str) -> str:
        if not self._compute_resource_name_to_id_cache:
            self.refresh_compute_resources()

        return self._compute_resource_name_to_id_cache[resource_name]

    def map_compute_resource_id_to_name(self, resource_id: str) -> str:
        if not self._compute_resource_id_to_name_cache:
            self.refresh_compute_resources()

        return self._compute_resource_id_to_name_cache.get(resource_id, '')

    def refresh_projects(self) -> None:
        self._project_name_to_id_cache.clear()
        self._project_id_to_name_cache.clear()

        try:
            projects = self._iri_client.account.get_projects()
        except Exception:
            logger.warning('Failed to fetch projects', exc_info=True)
        else:
            for project in projects:
                self._project_name_to_id_cache[project.name] = project.id
                self._project_id_to_name_cache[project.id] = project.name

        api_base_url = self._iri_client.get_api_base_url()
        logger.info(
            f'Fetched projects from {api_base_url}:\n'
            + json.dumps(self._project_name_to_id_cache, indent=2)
        )

    def project_names(self) -> Sequence[str]:
        if not self._project_name_to_id_cache:
            self.refresh_projects()

        return sorted(self._project_name_to_id_cache.keys())

    def map_project_name_to_id(self, project_name: str) -> str:
        if not self._project_name_to_id_cache:
            self.refresh_projects()

        return self._project_name_to_id_cache[project_name]

    def map_project_id_to_name(self, project_id: str) -> str:
        if not self._project_id_to_name_cache:
            self.refresh_projects()

        return self._project_id_to_name_cache.get(project_id, '')

    @abstractmethod
    def globus_collections(self) -> Mapping[str, GlobusCollection]:
        pass

    def get_default_globus_collection(self) -> GlobusCollection:
        return next(iter(self.globus_collections().values()))

    @abstractmethod
    def create_job_specification(
        self, action: str, input_directory: Path, output_directory: Path
    ) -> JobSpecification:
        pass


class ALCFFacilityAdapter(IRIFacilityAdapter):
    NAME: Final[str] = 'ALCF'

    def __init__(self, settings: GenesisSettings, access_token: str) -> None:
        iri_client = IRIClient('https://api.alcf.anl.gov/api/v1/', access_token)
        super().__init__(iri_client)
        self._settings = settings

    def globus_collections(self) -> Mapping[str, GlobusCollection]:
        return {
            'dtn_home': GlobusCollection(
                id=UUID('9032dd3a-e841-4687-a163-2720da731b5b'),
                globus_path='/~/',
                posix_path=Path('/home'),
            ),
            'dtn_eagle': GlobusCollection(
                id=UUID('05d2c76a-e867-4f67-aa57-76edeb0beda0'),
                globus_path='/~/',
                posix_path=Path('/eagle'),
            ),
            'dtn_grand': GlobusCollection(
                id=UUID('3caddd4a-bb35-4c3d-9101-d9a0ad7f3a30'),
                globus_path='/~/',
                posix_path=Path('/grand'),
            ),
        }

    def create_job_specification(
        self, action: str, input_directory: Path, output_directory: Path
    ) -> JobSpecification:
        settings_file = input_directory / StandardFileLayout.SETTINGS
        command_list = [
            'source $HOME/.local/bin/env',
            f'ptychodus -b {action} -i {input_directory} -o {output_directory} -s {settings_file}',
        ]
        commands = '; '.join(command.strip().replace('"', '\\"') for command in command_list)
        outputs_directory = output_directory / 'outputs'

        return JobSpecification(
            executable='/bin/bash',
            arguments=['-c', commands],
            name=f'ptychodus-{action}',
            stdout_path=str(outputs_directory),
            stderr_path=str(outputs_directory),
            resources=ResourceSpecification(
                node_count=1,
            ),
            attributes=JobAttributes(
                duration=300,
                queue_name=self._settings.queue_name.get_value(),
                account=self._settings.account.get_value(),
                custom_attributes={'filesystems': 'eagle'},
            ),
        )


class NERSCFacilityAdapter(IRIFacilityAdapter):
    NAME: Final[str] = 'NERSC'

    def __init__(self, settings: GenesisSettings, access_token: str) -> None:
        iri_client = IRIClient('https://api.iri.nersc.gov/api/v1/', access_token)
        super().__init__(iri_client)
        self._settings = settings

    def globus_collections(self) -> Mapping[str, GlobusCollection]:
        return {
            'DTN': GlobusCollection(
                id=UUID('9d6d994a-6d04-11e5-ba46-22000b92c6ec'),
                globus_path='/global/cfs/cdirs/m5074',
                posix_path=Path('/global/cfs/cdirs/m5074'),
            )
        }

    def create_job_specification(
        self, action: str, input_directory: Path, output_directory: Path
    ) -> JobSpecification:
        # FIXME implement
        return JobSpecification(
            executable='/bin/hostname',
            resources=None,
            attributes=None,
        )

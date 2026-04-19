from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Final
from uuid import UUID
import logging

from .iri import IRIClient, JobSpecification, ResourceType

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GlobusCollection:
    id: UUID
    globus_path: str
    posix_path: Path


class IRIFacilityAdapter(ABC):
    def __init__(self, iri_client: IRIClient, default_compute_resource_id: str) -> None:
        self._iri_client = iri_client
        self._compute_resource_name_to_id_cache: dict[str, str] = {}
        self._compute_resource_id_to_name_cache: dict[str, str] = {}
        self._default_compute_resource_id = default_compute_resource_id

    def get_iri_client(self) -> IRIClient:
        return self._iri_client

    def get_default_compute_resource_id(self) -> str:
        return self._default_compute_resource_id

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

    def __init__(self, access_token: str) -> None:
        # FIXME 'Polaris': '55c1c993-1124-47f9-b823-514ba3849a9a',
        iri_client = IRIClient('https://api.alcf.anl.gov/api/v1/', access_token)
        super().__init__(iri_client, '55c1c993-1124-47f9-b823-514ba3849a9a')

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
        # FIXME implement
        return JobSpecification(
            executable='/bin/hostname',
            resources=None,
            attributes=None,
        )


class NERSCFacilityAdapter(IRIFacilityAdapter):
    NAME: Final[str] = 'NERSC'

    def __init__(self, access_token: str) -> None:
        # FIXME 'Perlmutter': '94351904-6dba-4c16-b5cd-fbd280d8615b',
        iri_client = IRIClient('https://api.iri.nersc.gov/api/v1/', access_token)
        super().__init__(iri_client, '94351904-6dba-4c16-b5cd-fbd280d8615b')

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

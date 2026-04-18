from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from uuid import UUID
import logging

from .iri import IRIClient, JobSpecification

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GlobusCollection:
    id: UUID
    globus_path: str
    posix_path: Path


class IRIFacilityAdapter(ABC):
    @abstractmethod
    def get_iri_client(self) -> IRIClient:
        pass

    @abstractmethod
    def compute_resource_ids(self) -> Mapping[str, UUID]:
        pass

    def get_default_compute_resource_id(self) -> UUID:
        return next(iter(self.compute_resource_ids().values()))

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
        self._iri_client = IRIClient('https://api.alcf.anl.gov/api/v1/', access_token)

    def get_iri_client(self) -> IRIClient:
        return self._iri_client

    def compute_resource_ids(self) -> Mapping[str, UUID]:
        return {
            'Polaris': UUID('55c1c993-1124-47f9-b823-514ba3849a9a'),
        }

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
        self._iri_client = IRIClient('https://api.iri.nersc.gov/api/v1/', access_token)

    def get_iri_client(self) -> IRIClient:
        return self._iri_client

    def compute_resource_ids(self) -> Mapping[str, UUID]:
        return {
            'Perlmutter': UUID('94351904-6dba-4c16-b5cd-fbd280d8615b'),
        }

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

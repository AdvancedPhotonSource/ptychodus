from collections.abc import Mapping
from uuid import UUID

from pydantic import BaseModel


class GlobusCollection(BaseModel):
    id: UUID
    globus_base_path: str
    posix_base_path: str

    def get_url(self, subpath: str = '') -> str:
        return f'globus://{self.id}{self.globus_base_path}{subpath}'


class GenesisFacility(BaseModel):
    compute_api_base_url: str
    compute_resource_ids: Mapping[str, UUID]
    globus_collections: Mapping[str, GlobusCollection]


def get_genesis_facilities() -> Mapping[str, GenesisFacility]:
    return {
        'ALCF': GenesisFacility(
            compute_api_base_url='https://api.alcf.anl.gov/api/v1/',
            compute_resource_ids={
                'Polaris': UUID('55c1c993-1124-47f9-b823-514ba3849a9a'),
            },
            globus_collections={
                'dtn_home': GlobusCollection(
                    id=UUID('9032dd3a-e841-4687-a163-2720da731b5b'),
                    globus_base_path='/~/',
                    posix_base_path='/home',
                ),
                'dtn_eagle': GlobusCollection(
                    id=UUID('05d2c76a-e867-4f67-aa57-76edeb0beda0'),
                    globus_base_path='/~/',
                    posix_base_path='/eagle',
                ),
                'dtn_grand': GlobusCollection(
                    id=UUID('3caddd4a-bb35-4c3d-9101-d9a0ad7f3a30'),
                    globus_base_path='/~/',
                    posix_base_path='/grand',
                ),
            },
        ),
        'NERSC': GenesisFacility(
            compute_api_base_url='https://api.iri.nersc.gov/api/v1/',
            compute_resource_ids={
                'Perlmutter': UUID('94351904-6dba-4c16-b5cd-fbd280d8615b'),
            },
            globus_collections={
                'DTN': GlobusCollection(
                    id=UUID('9d6d994a-6d04-11e5-ba46-22000b92c6ec'),
                    globus_base_path='/global/cfs/cdirs/m5074',
                    posix_base_path='/global/cfs/cdirs/m5074',
                )
            },
        ),
    }

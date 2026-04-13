from collections.abc import Mapping, Sequence
from enum import StrEnum, auto
from typing import Any
import time
import logging

from pydantic import BaseModel, ConfigDict, Field, StrictBool
import requests

from .token_storage import (
    GenesisAccessTokens,
    create_headers,
    get_compute_access_tokens_file,
    write_access_tokens,
)

logger = logging.getLogger(__name__)


class ResourceSpecification(BaseModel):
    node_count: int | None = None
    process_count: int | None = None
    processes_per_node: int | None = None
    cpu_cores_per_process: int | None = None
    gpu_cores_per_process: int | None = None
    exclusive_node_use: StrictBool = True
    memory: int | None = None


class JobAttributes(BaseModel):
    duration: int | None = None
    queue_name: str | None = None
    account: str | None = None
    reservation_id: str | None = None
    custom_attributes: Mapping[str, str] = Field(default_factory=dict)


class ContainerVolumeMount(BaseModel):
    source: str
    target: str
    read_only: StrictBool = True


class Container(BaseModel):
    image: str
    volume_mounts: Sequence[ContainerVolumeMount] = Field(default_factory=list)


class JobSpecification(BaseModel):
    model_config = ConfigDict(extra='ignore')
    executable: str | None = None
    container: Container | None = None
    arguments: Sequence[str] = Field(default_factory=list)
    directory: str | None = None
    name: str | None = None
    inherit_environment: StrictBool = True
    environment: Mapping[str, str] = Field(default_factory=dict)
    stdin_path: str | None = None
    stdout_path: str | None = None
    stderr_path: str | None = None
    resources: ResourceSpecification | None = None
    attributes: JobAttributes | None = None
    pre_launch: str | None = None
    post_launch: str | None = None
    launcher: str | None = None


class JobState(StrEnum):
    NEW = auto()
    QUEUED = auto()
    HELD = auto()
    ACTIVE = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELED = auto()


class JobStatus(BaseModel):
    state: JobState
    time: float | None = None
    message: str | None = None
    exit_code: int | None = None
    meta_data: Mapping[str, Any] | None = None


class JobResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    job_id: str = Field(alias='id')
    status: JobStatus | None = None
    job_specification: JobSpecification | None = Field(None, alias='job_spec')


class GenesisComputeClient:
    # See https://api.iri.nersc.gov/#/compute

    def __init__(self, api_base_url: str, access_token: str) -> None:
        self._base_url = f'{api_base_url}/compute'
        self._headers = create_headers(access_token)

    def submit_job(self, resource_id: str, spec: JobSpecification) -> JobResponse:
        response = requests.post(
            f'{self._base_url}/job/{resource_id}',
            json=spec.model_dump(mode='json'),
            headers=self._headers,
        )
        response.raise_for_status()
        return JobResponse.model_validate(response.json())

    def submit_job_with_retry(
        self,
        resource_id: str,
        spec: JobSpecification,
        max_retries: int = 10,
        retry_delay: float = 1.0,
    ) -> JobResponse:
        for attempt in range(1, max_retries + 1):
            try:
                return self.submit_job(resource_id, spec)
            except requests.HTTPError as exc:
                if exc.response.status_code == 404:
                    if attempt == max_retries:
                        logger.warning('Exceeded max retries.')
                        raise

                    time.sleep(retry_delay)
                else:
                    raise

        raise RuntimeError('submit_job_with_retry exited loop without returning')

    def update_job(self, resource_id: str, job_id: str, spec: JobSpecification) -> JobResponse:
        response = requests.put(
            f'{self._base_url}/job/{resource_id}/{job_id}',
            json=spec.model_dump(mode='json'),
            headers=self._headers,
        )
        response.raise_for_status()
        return JobResponse.model_validate(response.json())

    def get_job_status(
        self,
        resource_id: str,
        job_id: str,
        historical: bool = False,
        include_spec: bool = False,
    ) -> JobResponse:
        response = requests.get(
            f'{self._base_url}/status/{resource_id}/{job_id}',
            params={'historical': historical, 'include_spec': include_spec},
            headers=self._headers,
        )
        response.raise_for_status()
        return JobResponse.model_validate(response.json())

    def get_job_statuses(
        self,
        resource_id: str,
        offset: int = 0,
        limit: int = 100,
        historical: bool = False,
        include_spec: bool = False,
    ) -> Sequence[JobResponse]:
        response = requests.post(
            f'{self._base_url}/status/{resource_id}',
            params={
                'offset': offset,
                'limit': limit,
                'historical': historical,
                'include_spec': include_spec,
            },
            headers=self._headers,
        )
        response.raise_for_status()
        return [JobResponse.model_validate(item) for item in response.json()]

    def cancel_job(self, resource_id: str, job_id: str) -> bool:
        response = requests.delete(
            f'{self._base_url}/cancel/{resource_id}/{job_id}', headers=self._headers
        )
        response.raise_for_status()
        return response.status_code == 204


if __name__ == '__main__':
    tokens_file = get_compute_access_tokens_file()
    access_tokens: list[GenesisAccessTokens] = []

    while True:
        name = input('Enter a name for the access token (or blank to finish): ').strip()

        if not name:
            break

        api_base_url = input('Enter the API base URL: ').strip()
        access_token = input('Enter the access token: ').strip()
        access_tokens.append(
            GenesisAccessTokens(
                name=name,
                api_base_url=api_base_url,
                access_token=access_token,
            )
        )

    write_access_tokens(tokens_file, access_tokens)

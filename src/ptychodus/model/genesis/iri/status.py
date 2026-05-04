from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum, auto
import logging

from pydantic import BaseModel
import requests

from ..tokens import create_headers

logger = logging.getLogger(__name__)


class ResourceType(StrEnum):
    WEBSITE = auto()
    SERVICE = auto()
    COMPUTE = auto()
    SYSTEM = auto()
    STORAGE = auto()
    NETWORK = auto()
    UNKNOWN = auto()


class ResourceStatus(StrEnum):
    UP = auto()
    DOWN = auto()
    DEGRADED = auto()
    UNKNOWN = auto()


class AllocationUnit(StrEnum):
    NODE_HOURS = auto()
    BYTES = auto()
    INODES = auto()


class Resource(BaseModel):
    id: str
    name: str | None = None
    description: str | None = None
    last_modified: datetime
    group: str | None = None
    current_status: ResourceStatus | None = None
    resource_type: ResourceType
    self_uri: str
    site_uri: str
    capability_uris: list[str]


class IRIStatusClient:
    # See https://api.iri.nersc.gov/#/status

    def __init__(self, api_base_url: str, access_token: str) -> None:
        self._base_url = api_base_url.rstrip('/') + '/api/v1/status'
        self._headers = create_headers(access_token)

    def get_resources(
        self,
        name: str | None = None,
        description: str | None = None,
        group: str | None = None,
        offset: int = 0,
        limit: int = 100,
        modified_since: datetime | None = None,
        resource_type: ResourceType | None = None,
        current_status: ResourceStatus | None = None,
        capability: Sequence[AllocationUnit] | None = None,
    ) -> Sequence[Resource]:
        params: dict = {'offset': offset, 'limit': limit}
        if name is not None:
            params['name'] = name
        if description is not None:
            params['description'] = description
        if group is not None:
            params['group'] = group
        if modified_since is not None:
            params['modified_since'] = modified_since.isoformat()
        if resource_type is not None:
            params['resource_type'] = resource_type
        if current_status is not None:
            params['current_status'] = current_status
        if capability is not None:
            params['capability'] = [c.value for c in capability]
        response = requests.get(
            f'{self._base_url}/resources',
            params=params,
            headers=self._headers,
        )
        response.raise_for_status()
        return [Resource.model_validate(item) for item in response.json()]

    def get_resource(self, resource_id: str) -> Resource:
        response = requests.get(
            f'{self._base_url}/resources/{resource_id}',
            headers=self._headers,
        )
        response.raise_for_status()
        return Resource.model_validate(response.json())

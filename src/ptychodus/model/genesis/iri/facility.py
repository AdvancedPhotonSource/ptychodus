from collections.abc import Sequence
from datetime import datetime
import logging

from pydantic import BaseModel
import requests

from ..tokens import create_headers

logger = logging.getLogger(__name__)


class Facility(BaseModel):
    id: str
    name: str | None = None
    description: str | None = None
    last_modified: datetime
    short_name: str | None = None
    organization_name: str | None = None
    support_uri: str | None = None
    self_uri: str
    site_uris: list[str]


class Site(BaseModel):
    id: str
    name: str | None = None
    description: str | None = None
    last_modified: datetime
    short_name: str | None = None
    operating_organization: str | None = None
    country_name: str | None = None
    locality_name: str | None = None
    state_or_province_name: str | None = None
    street_address: str | None = None
    unlocode: str | None = None
    altitude: float | None = None
    latitude: float | None = None
    longitude: float | None = None
    self_uri: str
    resource_uris: list[str]


class IRIFacilityClient:
    # See https://api.iri.nersc.gov/#/facility

    def __init__(self, api_base_url: str, access_token: str) -> None:
        self._base_url = api_base_url.rstrip('/') + '/api/v1/facility'
        self._headers = create_headers(access_token)

    def get_facility(self, modified_since: datetime | None = None) -> Facility:
        params: dict = {}

        if modified_since is not None:
            params['modified_since'] = modified_since.isoformat()

        response = requests.get(
            self._base_url,
            params=params,
            headers=self._headers,
        )
        response.raise_for_status()
        return Facility.model_validate(response.json())

    def get_sites(
        self,
        modified_since: datetime | None = None,
        name: str | None = None,
        offset: int = 0,
        limit: int = 100,
        short_name: str | None = None,
    ) -> Sequence[Site]:
        params: dict = {'offset': offset, 'limit': limit}

        if modified_since is not None:
            params['modified_since'] = modified_since.isoformat()

        if name is not None:
            params['name'] = name

        if short_name is not None:
            params['short_name'] = short_name

        response = requests.get(
            f'{self._base_url}/sites',
            params=params,
            headers=self._headers,
        )
        response.raise_for_status()
        return [Site.model_validate(item) for item in response.json()]

    def get_site(self, site_id: str, modified_since: datetime | None = None) -> Site:
        params: dict = {}

        if modified_since is not None:
            params['modified_since'] = modified_since.isoformat()

        response = requests.get(
            f'{self._base_url}/sites/{site_id}',
            params=params,
            headers=self._headers,
        )
        response.raise_for_status()
        return Site.model_validate(response.json())

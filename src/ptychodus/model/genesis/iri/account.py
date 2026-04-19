from collections.abc import Sequence
from datetime import datetime
import logging

from pydantic import BaseModel
import requests

from ..tokens import create_headers

logger = logging.getLogger(__name__)


class Project(BaseModel):
    id: str
    name: str
    description: str
    user_ids: Sequence[str]
    last_modified: datetime
    self_uri: str


class IRIAccountClient:
    # See https://api.iri.nersc.gov/#/account

    def __init__(self, api_base_url: str, access_token: str) -> None:
        self._base_url = api_base_url.rstrip('/') + '/account'
        self._headers = create_headers(access_token)

    def get_projects(self) -> Sequence[Project]:
        response = requests.get(
            f'{self._base_url}/projects',
            headers=self._headers,
        )
        response.raise_for_status()
        return [Project.model_validate(item) for item in response.json()]

    def get_project(self, project_id: str) -> Project:
        response = requests.get(
            f'{self._base_url}/projects/{project_id}',
            headers=self._headers,
        )
        response.raise_for_status()
        return Project.model_validate(response.json())

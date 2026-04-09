import json
import os
import stat
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel


def create_headers(access_token: str) -> Mapping[str, str]:
    return {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json',
    }


class GenesisAccessTokens(BaseModel):
    transfer_access_token: str
    compute_access_token: str


def read_access_tokens(file_path: Path) -> Mapping[str, GenesisAccessTokens]:
    mode = file_path.stat().st_mode

    if mode & 0o177:
        raise PermissionError(
            f'{file_path} has insecure permissions {stat.filemode(mode)}; '
            f'expected 0o600 (owner read/write only)'
        )

    with open(file_path) as f:
        data = json.load(f)

    return {name: GenesisAccessTokens.model_validate(entry) for name, entry in data.items()}


def write_access_tokens(file_path: Path, entries: Mapping[str, GenesisAccessTokens]) -> None:
    data = {name: entry.model_dump(mode='json') for name, entry in entries.items()}
    fd = os.open(file_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)

    with open(fd, 'w') as f:
        json.dump(data, f, indent=2)

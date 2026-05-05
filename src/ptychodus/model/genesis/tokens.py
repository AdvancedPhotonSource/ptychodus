from collections.abc import Mapping, Sequence
from pathlib import Path
import json
import os
import stat

from pydantic import BaseModel


def create_headers(access_token: str) -> Mapping[str, str]:
    return {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json',
    }


class GenesisAccessTokens(BaseModel):
    facility: str
    access_token: str


def load_tokens(file_path: Path) -> Sequence[GenesisAccessTokens]:
    mode = file_path.stat().st_mode

    if mode & 0o177:
        raise PermissionError(
            f'{file_path} has insecure permissions {stat.filemode(mode)}; '
            f'expected 0o600 (owner read/write only)'
        )

    with open(file_path) as f:
        data = json.load(f)

    return [GenesisAccessTokens.model_validate(token) for token in data]


def save_tokens(file_path: Path, access_tokens: Sequence[GenesisAccessTokens]) -> None:
    data = [token.model_dump(mode='json') for token in access_tokens]
    fd = os.open(file_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)

    with open(fd, 'w') as f:
        json.dump(data, f, indent=2)

"""Bearer tokens for the MCP server: a file of hashed client tokens, and their verifier.

The token file is a JSON list of ``{"client_id": ..., "token_sha256": ...}`` entries, readable
by its owner alone. It holds each token's SHA-256 digest rather than the token, so the file
grants nothing by itself; a token is shown once, when it is minted.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import stat
import tempfile
from collections.abc import Sequence
from pathlib import Path

from fastmcp.server.auth import AccessToken, TokenVerifier
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

__all__ = [
    'ClientToken',
    'ClientTokenVerifier',
    'load_client_tokens',
    'mint_client_token',
    'revoke_client_token',
]


class ClientToken(BaseModel):
    """One MCP client and the SHA-256 digest of its bearer token."""

    model_config = ConfigDict(extra='forbid')

    client_id: str = Field(min_length=1)
    token_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def load_client_tokens(file_path: Path) -> list[ClientToken]:
    """Read a token file.

    Raises:
        PermissionError: If anyone but the owner may read or write the file.
        ValueError: If the file is malformed or names a client twice.
    """
    mode = file_path.stat().st_mode

    if mode & 0o177:
        raise PermissionError(
            f'{file_path} has insecure permissions {stat.filemode(mode)}; '
            f'expected 0o600 (owner read/write only)'
        )

    tokens = TypeAdapter(list[ClientToken]).validate_json(file_path.read_bytes())
    client_ids = [token.client_id for token in tokens]
    duplicates = sorted({client_id for client_id in client_ids if client_ids.count(client_id) > 1})

    if duplicates:
        raise ValueError(f'{file_path} names clients more than once: {", ".join(duplicates)}')

    return tokens


def _save_client_tokens(file_path: Path, tokens: Sequence[ClientToken]) -> None:
    """Replace the token file atomically, readable by its owner alone."""
    data = json.dumps([token.model_dump(mode='json') for token in tokens], indent=2)
    file_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    # mkstemp creates a file of its own under a fresh name, exclusively and readable by its
    # owner alone, so neither a planted file nor one left by an interrupted write is reused.
    fd, temporary_name = tempfile.mkstemp(dir=file_path.parent, prefix=f'.{file_path.name}.')
    temporary_path = Path(temporary_name)

    try:
        with open(fd, 'w') as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())

        os.replace(temporary_path, file_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def mint_client_token(file_path: Path, client_id: str) -> str:
    """Make a new bearer token for `client_id`, record its digest, and return the token.

    A client already in the file gets a new token, and its old one stops working once the
    webapp restarts. The file is created if it does not exist.
    """
    tokens = load_client_tokens(file_path) if file_path.exists() else []
    token = secrets.token_urlsafe(32)
    entry = ClientToken(client_id=client_id, token_sha256=_hash_token(token))
    kept = [existing for existing in tokens if existing.client_id != client_id]
    _save_client_tokens(file_path, [*kept, entry])
    return token


def revoke_client_token(file_path: Path, client_id: str) -> None:
    """Remove `client_id` from the token file.

    Raises:
        LookupError: If the file does not name `client_id`.
    """
    tokens = load_client_tokens(file_path)
    kept = [token for token in tokens if token.client_id != client_id]

    if len(kept) == len(tokens):
        raise LookupError(f'{file_path} names no client {client_id!r}')

    _save_client_tokens(file_path, kept)


class ClientTokenVerifier(TokenVerifier):
    """Accepts a bearer token whose SHA-256 digest is in the token file, as that client."""

    def __init__(self, tokens: Sequence[ClientToken]) -> None:
        super().__init__()
        self._tokens = list(tokens)

    @property
    def client_ids(self) -> list[str]:
        return [token.client_id for token in self._tokens]

    async def verify_token(self, token: str) -> AccessToken | None:
        digest = _hash_token(token)
        match: ClientToken | None = None

        # Every entry is compared, in constant time, so the time taken reveals nothing.
        for entry in self._tokens:
            if hmac.compare_digest(digest, entry.token_sha256):
                match = entry

        if match is None:
            return None

        return AccessToken(token=token, client_id=match.client_id, scopes=[])

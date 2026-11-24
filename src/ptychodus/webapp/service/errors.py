"""The refusals a service operation raises, each mapped by a transport to its own shape."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

__all__ = ['InvalidRequestError', 'ResourceNotFoundError']


class ResourceNotFoundError(LookupError):
    """What was asked for does not exist: an unknown uuid, or an index out of range.

    `available` names the choices there were, where they have names.
    """

    def __init__(self, message: str, *, available: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.message = message
        self.available = list(available)


class InvalidRequestError(ValueError):
    """The request cannot be served as asked: a bad parameter, or data a report cannot use.

    `detail` is structured detail beside the message, such as the valid choices.
    """

    def __init__(self, message: str, *, detail: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = dict(detail) if detail is not None else None

"""The operation classes a service operation declares, and how a transport reads them back."""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from functools import partial
from typing import Any, TypeVar

__all__ = ['OperationClass', 'operation', 'operation_class_of']

_F = TypeVar('_F', bound=Callable[..., Any])


class OperationClass(StrEnum):
    """How long an operation takes, and so how it runs.

    An instant operation reads the metadata cache and answers inline. A bounded one also
    reads a resource's data and computes on it, in a worker thread under the shared
    concurrency cap, and answers within one request.
    """

    INSTANT = 'instant'
    BOUNDED = 'bounded'


def _mark(op_class: OperationClass, func: _F) -> _F:
    func.__operation_class__ = op_class  # type: ignore[attr-defined]
    return func


def operation(op_class: OperationClass) -> Callable[[_F], _F]:
    """Decorate a service method with its operation class."""
    return partial(_mark, op_class)


def operation_class_of(func: Callable[..., Any]) -> OperationClass:
    """The class `func` declared with :func:`operation`.

    Raises:
        TypeError: If `func` declared none.
    """
    op_class = getattr(func, '__operation_class__', None)

    if not isinstance(op_class, OperationClass):
        raise TypeError(f'{func.__qualname__} declares no operation class.')

    return op_class

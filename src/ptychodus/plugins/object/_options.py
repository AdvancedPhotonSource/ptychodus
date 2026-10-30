"""Checks shared by every object generator in this package."""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

OptionsT = TypeVar('OptionsT', bound=BaseModel)


def expect_options(options: BaseModel, kind: type[OptionsT]) -> OptionsT:
    """Return *options* as *kind*, refusing options declared for another generator."""
    if not isinstance(options, kind):
        raise TypeError(f'Expected {kind.__name__}, got {type(options).__name__}.')

    return options

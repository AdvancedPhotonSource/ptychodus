"""Field declarations and checks shared by every probe generator in this package."""

from __future__ import annotations

from typing import Any, TypeVar

from pydantic import BaseModel, Field

OptionsT = TypeVar('OptionsT', bound=BaseModel)


def defocus_distance_field() -> Any:
    """The object plane's position past the focus.

    Every generator declaring it persists it under one key, so all of them must declare
    it identically; declaring it once here is what makes them.
    """
    return Field(
        0.0,
        title='Defocus Distance:',
        description=(
            'How far past its focus the object sits: positive downstream (diverging), '
            "negative upstream (converging). The product's focus wins when it states one."
        ),
        json_schema_extra={'display_unit': 'MICROMETER'},
    )


def propagation_distance_field() -> Any:
    """How far an already formed field is carried, independent of the defocus.

    Shared by key for the same reason as :func:`defocus_distance_field`.
    """
    return Field(
        0.0,
        title='Propagation Distance:',
        description=(
            'Moves the plane of the formed field, after any defocus curvature, without '
            'changing the defocus.'
        ),
        json_schema_extra={'display_unit': 'MICROMETER'},
    )


def expect_options(options: BaseModel, kind: type[OptionsT]) -> OptionsT:
    """Return *options* as *kind*, refusing options declared for another generator."""
    if not isinstance(options, kind):
        raise TypeError(f'Expected {kind.__name__}, got {type(options).__name__}.')

    return options

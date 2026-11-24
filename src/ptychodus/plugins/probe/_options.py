"""Field declarations shared by every probe generator in this package."""

from __future__ import annotations

from typing import Any

from pydantic import Field


def propagation_distance_field() -> Any:
    """How far an already formed field is carried, independent of the defocus.

    Every generator declaring it persists it under one key, so all of them must declare
    it identically; declaring it once here is what makes them.
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

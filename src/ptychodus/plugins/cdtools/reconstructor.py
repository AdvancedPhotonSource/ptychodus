"""Parent-side factory that builds the :class:`SubprocessReconstructor` for cdtools.

Zero torch imports. All GPU work runs inside a spawned child; see :mod:`._subprocess`.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from ptychodus.api.reconstruct import ReconstructInput, SubprocessReconstructor

from .geometry import verify_product
from .options import CDToolsOptions, default_options

__all__ = [
    'build_reconstructor',
]


def _declared(options: BaseModel) -> CDToolsOptions:
    """Narrow the options a call was given to what this backend declares.

    Checked rather than asserted: the dispatching layer validates the type against what
    ``get_default_options`` returns, so a mismatch here means those two have drifted
    apart, and saying so beats an attribute error several frames later.
    """
    if not isinstance(options, CDToolsOptions):
        raise TypeError(f'Expected CDToolsOptions, got {type(options).__name__}.')

    return options


def build_reconstructor() -> SubprocessReconstructor:
    """Build the :class:`SubprocessReconstructor` for cdtools' FancyPtycho model."""

    def dump_reconstruct_options(input_: ReconstructInput, options: BaseModel) -> dict[str, Any]:
        declared = _declared(options)

        # Checked here so a product FancyPtycho cannot take is refused before a child is
        # spawned, and so is a schedule with nothing in it.
        verify_product(input_.product)

        if not declared.schedule.stages():
            raise ValueError('Every cdtools stage has zero epochs; there is nothing to run.')

        return {'options': declared.model_copy(deep=True)}

    return SubprocessReconstructor(
        name='FancyPtycho',
        default_options=default_options,
        dump_reconstruct_options=dump_reconstruct_options,
    )

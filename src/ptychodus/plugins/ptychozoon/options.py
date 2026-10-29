"""What the ptychozoon enhancer accepts.

The field names are upstream's own, from ``ptychozoon.settings.LSMRSettings``, so
each maps onto the setting of the same name in the child.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = [
    'PtychozoonOptions',
]


class PtychozoonOptions(BaseModel):
    damping_factor: float = Field(
        0.0,
        ge=0.0,
        title='Damping Factor:',
        description='Tikhonov regularization of the least-squares solve; zero applies none.',
    )
    max_iter: int = Field(
        100,
        ge=1,
        title='Max Iterations:',
        description='Total number of LSMR iterations.',
    )
    atol: float = Field(
        1e-6,
        ge=0.0,
        title='A Tolerance:',
        description='LSMR stops once the relative residual falls below this.',
    )
    btol: float = Field(
        1e-6,
        ge=0.0,
        title='B Tolerance:',
        description='LSMR stops once the relative residual of the normal equations falls below this.',
    )
    checkpoint_interval: int = Field(
        5,
        ge=1,
        title='Checkpoint Interval:',
        description='Report the solution every this many iterations.',
    )

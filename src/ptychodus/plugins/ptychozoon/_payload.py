"""What crosses between the ptychozoon enhancer and its child.

Only ptychodus types: the child builds the ptychozoon objects itself, so the parent
never imports ptychozoon at all. Both sides import this module, since the entry
point is addressed by name and the parent must not import the child.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ptychodus.api.fluorescence import FluorescenceEnhancerInput

from .options import PtychozoonOptions

__all__ = [
    'PtychozoonMessage',
    'PtychozoonPayload',
]


class PtychozoonMessage(StrEnum):
    """What the enhancement child has to say about its work.

    The transport owns the tags it puts on the queue itself and passes every
    other one through, so the vocabulary of an enhancement run is named here.

    A string enum because the value is what travels, so a child writing the
    bare string still compares equal to the member.
    """

    OUTPUT = 'output'
    """One checkpoint, as a :class:`~ptychodus.api.fluorescence.FluorescenceEnhancerOutput`."""


@dataclass(frozen=True)
class PtychozoonPayload:
    """Inputs for one ptychozoon enhancement run."""

    input_: FluorescenceEnhancerInput
    options: PtychozoonOptions

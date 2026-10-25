"""Where each Kickstart option is persisted in ptychodus's flat settings layout.

Kickstart's declaration is ptychodus's own and was written in house style from the
start, so the keys follow the rules -- PascalCase with units spelled out, enums by
value -- and the whole declaration is one group. **Transitional**, with the flat
layout it serves.
"""

from __future__ import annotations

from collections.abc import Sequence

from .flatten import EnumSpelling, FlatteningPolicy, NamingConvention
from .parameter_store import OptionsSection

__all__ = ['sections']


def sections() -> Sequence[OptionsSection]:
    """Return the one settings group backing Kickstart, the whole declaration."""
    policy = FlatteningPolicy(naming=NamingConvention.PASCAL_CASE, spelling=EnumSpelling.BY_VALUE)

    return [
        OptionsSection('KickstartOptions', policy=policy),
    ]

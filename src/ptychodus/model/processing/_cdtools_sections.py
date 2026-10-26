"""Where each cdtools option is persisted in ptychodus's flat settings layout.

The cdtools declaration is ptychodus's own and was written in house style from the
start, so the keys follow the rules -- PascalCase with units spelled out, enums by
value -- and only the mapping from declared branch to group lives here. Each stage of
the schedule contributes its own prefix, so the schedule's keys read ``Stage1Epochs``
and so on. **Transitional**, with the flat layout it serves.
"""

from __future__ import annotations

from collections.abc import Sequence

from .flatten import EnumSpelling, FlatteningPolicy, NamingConvention
from .parameter_store import OptionsSection

__all__ = ['sections']


def sections() -> Sequence[OptionsSection]:
    """Return the four settings groups backing cdtools, one per declared branch."""
    policy = FlatteningPolicy(naming=NamingConvention.PASCAL_CASE, spelling=EnumSpelling.BY_VALUE)

    return [
        OptionsSection('CDToolsOptimizer', root_path='optimizer', policy=policy),
        OptionsSection('CDToolsSchedule', root_path='schedule', policy=policy),
        OptionsSection('CDToolsRefinement', root_path='refinement', policy=policy),
        OptionsSection('CDToolsFancyPtycho', root_path='model', policy=policy),
    ]

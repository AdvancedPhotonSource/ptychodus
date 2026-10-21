"""Where each PtychoPINN option is persisted in ptychodus's flat settings layout.

Thinner than pty-chi's table, because the declaration is ptychodus's own: it
states its bounds, choices and display text inline, so only the mapping from
declared path to persisted key lives here.

ptychodus persists PtychoPINN keys as the literal field names, so the policy is
verbatim and each branch simply names its group. **Transitional**, with the flat
layout it serves.
"""

from __future__ import annotations

from collections.abc import Sequence

from .flatten import EnumSpelling, FlatteningPolicy, NamingConvention
from .parameter_store import OptionsSection

__all__ = ['sections']


def sections() -> Sequence[OptionsSection]:
    """Return the four settings groups backing PtychoPINN.

    One per declared branch, with no prefix and no renaming: the persisted key
    is the field name. When the keys are normalized to house style, this becomes
    a `NamingConvention.PASCAL_CASE` policy plus whatever the rules then miss.
    """
    verbatim = FlatteningPolicy(naming=NamingConvention.VERBATIM, spelling=EnumSpelling.BY_VALUE)

    return [
        OptionsSection('PtychoPINNData', root_path='data', policy=verbatim),
        OptionsSection('PtychoPINNModel', root_path='model', policy=verbatim),
        OptionsSection('PtychoPINNTraining', root_path='training', policy=verbatim),
        OptionsSection('PtychoPINNInference', root_path='inference', policy=verbatim),
    ]

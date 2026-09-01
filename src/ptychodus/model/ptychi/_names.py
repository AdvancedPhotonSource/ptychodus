"""Display names for pty-chi algorithms, kept in a ptychi-free module.

algorithms.py imports ``ptychi.api`` at module scope, so anything downstream of
that import can't be reached when pty-chi is not installed. core.py's
developer-mode fallback path needs the display-name list, and algorithms.py's
_Spec entries also need to agree with it; both import from here to keep them
locked together without either side reaching across the ptychi guard.
"""

from __future__ import annotations

DISPLAY_NAMES: tuple[str, ...] = (
    'DM',
    'RAAR',
    'PIE',
    'ePIE',
    'rPIE',
    'LSQML',
    'Autodiff',
    'BH',
)

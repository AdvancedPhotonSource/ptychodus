"""Payload dataclass for the PtyChi subprocess entry point.

Parent-safe to import: pulls ``PtychographyTaskOptions`` from ``ptychi.api``,
which imports torch for its type annotations but does not acquire a GPU
context — that happens only when the child instantiates ``PtychographyTask``.
"""

from __future__ import annotations

from dataclasses import dataclass

from ptychi.api.options.task import PtychographyTaskOptions

from ptychodus.api.reconstructor import ReconstructInput


@dataclass(frozen=True)
class PtyChiPayload:
    """Everything the child needs to acquire a GPU context and run one reconstruction.

    ``task_options`` is a fully-populated algorithm-specific subclass of
    :class:`PtychographyTaskOptions` (``DMOptions``, ``PIEOptions``,
    ``LSQMLOptions``, ...) that the parent built via the option-helper classes
    in :mod:`.helper` and the per-algorithm classes in :mod:`.dm`, :mod:`.pie`,
    etc. It carries all diffraction data, positions, and initial guesses; the
    child hands it straight to ``PtychographyTask``.
    """

    task_options: PtychographyTaskOptions
    num_sync_epochs: int
    reconstruct_input: ReconstructInput

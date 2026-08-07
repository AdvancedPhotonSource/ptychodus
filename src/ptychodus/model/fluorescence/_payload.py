"""Payload dataclass for the ptychozoon VSPI subprocess entry point.

``ptychozoon`` is an optional extra, so it is imported under
:data:`~typing.TYPE_CHECKING` only: this module has ``from __future__ import
annotations`` and :class:`PtychozoonPayload` is a dataclass, whose field
annotations are never evaluated. That keeps ``ptychodus.model`` importable
without the extra installed. The enhancer defers the matching runtime imports
into :meth:`~.ptychozoon.PtychozoonFluorescenceEnhancer._build_payload`.

Only ``ptychozoon.data_structures`` and ``ptychozoon.settings`` may be reached
from the parent: they pull in just ``numpy``, ``dataclasses``, ``enum``, and
``typing`` (no CuPy). Anything else under ``ptychozoon.*`` is GPU-tainted and
must stay inside the child; see
[tests/test_no_gpu_context.py](../../../../tests/test_no_gpu_context.py) for
the allow-list that pins that guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ptychozoon.data_structures import (
        FluorescenceDataset,
        PtychographyProduct,
    )
    from ptychozoon.settings import DeconvolutionEnhancementSettings

__all__ = [
    'PtychozoonPayload',
]


@dataclass(frozen=True)
class PtychozoonPayload:
    """Picklable inputs for one ptychozoon enhancement run."""

    product: PtychographyProduct
    dataset: FluorescenceDataset
    settings: DeconvolutionEnhancementSettings

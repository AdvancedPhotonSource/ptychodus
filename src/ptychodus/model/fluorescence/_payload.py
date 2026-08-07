"""Payload dataclass for the ptychozoon VSPI subprocess entry point.

Parent-safe: ``ptychozoon.data_structures`` and ``ptychozoon.settings`` only
pull in ``numpy``, ``dataclasses``, ``enum``, and ``typing`` at import time
(no CuPy). Anything else under ``ptychozoon.*`` is GPU-tainted and must stay
inside the child; see
[tests/test_no_gpu_context.py](../../../../tests/test_no_gpu_context.py) for
the allow-list that pins that guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass

from ptychozoon.data_structures import (
    ElementMap,
    FluorescenceDataset,
    PtychographyProduct,
)
from ptychozoon.settings import (
    DeconvolutionEnhancementSettings,
    InterpolationTypes,
)

__all__ = [
    'DeconvolutionEnhancementSettings',
    'ElementMap',
    'FluorescenceDataset',
    'InterpolationTypes',
    'PtychographyProduct',
    'PtychozoonPayload',
]


@dataclass(frozen=True)
class PtychozoonPayload:
    """Picklable inputs for one ptychozoon enhancement run."""

    product: PtychographyProduct
    dataset: FluorescenceDataset
    settings: DeconvolutionEnhancementSettings

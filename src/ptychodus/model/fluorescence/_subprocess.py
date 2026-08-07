"""Subprocess worker that runs the ptychozoon (GPU VSPI) fluorescence enhancement.

This module is executed in a freshly ``spawn``ed process so that the CuPy GPU
context is created cleanly and fully released when the process exits. The
GPU-tainted ``ptychozoon.vspi_enhance`` import is deferred to the worker body
so the parent process never pulls CuPy into its own interpreter.

The payload carries fully constructed ``ptychozoon`` objects (see
:mod:`._payload`); this file just hands them to the enhancement algorithm and
streams checkpoint results back. Log forwarding and error marshaling are
handled by the shared subprocess protocol in
:mod:`ptychodus.model.processing._subprocess_protocol`.
"""

from __future__ import annotations

from multiprocessing.queues import Queue
from typing import Any

import numpy

from ._payload import PtychozoonPayload

__all__ = [
    'run_vspi_enhancement',
]


def run_vspi_enhancement(payload: PtychozoonPayload, result_queue: 'Queue[Any]') -> None:
    """Run ptychozoon VSPI enhancement and stream checkpoint results over a queue.

    Intended as the ``entry_point`` for
    :func:`ptychodus.model.processing._subprocess_protocol.run_subprocess`.
    Puts ``('result', iteration, [(name, cps), ...])`` tuples per checkpoint.
    """
    from ptychozoon.vspi_enhance import VSPIFluorescenceEnhancingAlgorithm

    algorithm = VSPIFluorescenceEnhancingAlgorithm()

    for enhanced_dataset, iteration in algorithm.enhance(
        payload.dataset, payload.product, settings=payload.settings
    ):
        result_queue.put(
            (
                'result',
                int(iteration),
                [
                    (emap.name, numpy.asarray(emap.counts_per_second))
                    for emap in enhanced_dataset.element_maps
                ],
            )
        )

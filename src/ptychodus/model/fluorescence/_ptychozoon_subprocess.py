"""Subprocess worker that runs the ptychozoon (GPU VSPI) fluorescence enhancement.

This module is executed in a freshly ``spawn``ed process so that the CuPy GPU
context is created cleanly and fully released when the process exits. It must not
import ``ptychozoon`` (or CuPy) at module load time; the import happens inside the
worker function so that the parent process never pulls GPU libraries into its own
interpreter.

All data crosses the process boundary as plain numpy arrays and scalars (see
:class:`PtychozoonPayload`). Messages are streamed back over a single queue as
tagged tuples:

- ``('result', iteration, [(element_name, counts_per_second_array), ...])`` per checkpoint
- ``('log', levelno, message)`` for each captured log record from the ptychozoon logger
- ``('error', traceback_str)`` on failure

followed by a ``None`` sentinel. Because a single child thread produces every
message, ordering is preserved (a log line arrives before the result it precedes).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from multiprocessing.queues import Queue
from typing import Any

import numpy

__all__ = [
    'PtychozoonPayload',
    'run_vspi_enhancement',
]


class _QueueLogHandler(logging.Handler):
    """Logging handler that forwards formatted records to the parent over the result queue."""

    def __init__(self, result_queue: 'Queue[Any]') -> None:
        super().__init__()
        self._result_queue = result_queue

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._result_queue.put(('log', record.levelno, self.format(record)))
        except Exception:
            # Never let logging failures break the enhancement worker.
            pass


@dataclass(frozen=True)
class PtychozoonPayload:
    """Picklable inputs for one ptychozoon enhancement run."""

    # Ptychography product (plain arrays / scalars)
    probe_positions_m: numpy.ndarray  # (N, 2) float [y, x] meters
    probe: numpy.ndarray  # (n_opr, modes, height, width) complex
    object_array: numpy.ndarray  # (height, width) complex
    pixel_size_m: tuple[float, float]  # (pixel_height_m, pixel_width_m)
    object_center_m: tuple[float, float]  # (center_y_m, center_x_m)
    opr_mode_weights: numpy.ndarray | None  # (n_opr, N) float or None

    # Fluorescence element maps
    element_maps: list[tuple[str, numpy.ndarray]]  # [(name, counts_per_second), ...]

    # Solver settings
    damping_factor: float
    gradient_smoothness: float
    max_iterations: int
    atol: float
    btol: float
    checkpoint_interval: int
    use_gpu: bool
    gpu_device_index: int

    # Effective log level of the parent's ptychozoon logger, so the child only
    # forwards records that the parent would actually surface.
    log_level: int


def run_vspi_enhancement(payload: PtychozoonPayload, result_queue: 'Queue[Any]') -> None:
    """Run ptychozoon VSPI enhancement and stream checkpoint results over a queue.

    Intended as the target of a ``spawn``ed ``multiprocessing.Process``. Puts
    ``('result', iteration, [(name, cps), ...])`` tuples per checkpoint and
    ``('log', levelno, message)`` tuples for ptychozoon log records, then ``None``
    when complete, or ``('error', traceback_str)`` on failure.
    """
    # Forward ptychozoon's own log output to the parent so it appears in the
    # Enhance Fluorescence status view (this process has no ptychodus handlers).
    log_handler = _QueueLogHandler(result_queue)
    log_handler.setFormatter(logging.Formatter('%(message)s'))
    ptychozoon_logger = logging.getLogger('ptychozoon')
    ptychozoon_logger.addHandler(log_handler)
    ptychozoon_logger.setLevel(payload.log_level)

    try:
        from ptychozoon.data_structures import (
            ElementMap,
            FluorescenceDataset,
            PtychographyProduct,
        )
        from ptychozoon.settings import (
            DeconvolutionEnhancementSettings,
            InterpolationTypes,
        )
        from ptychozoon.vspi_enhance import VSPIFluorescenceEnhancingAlgorithm

        product = PtychographyProduct(
            probe_positions=payload.probe_positions_m,
            probe=payload.probe,
            object_array=payload.object_array,
            pixel_size_m=payload.pixel_size_m,
            object_center_m=payload.object_center_m,
            opr_mode_weights=payload.opr_mode_weights,
        )
        dataset = FluorescenceDataset(
            element_maps=[
                ElementMap(name=name, counts_per_second=cps) for name, cps in payload.element_maps
            ]
        )

        settings = DeconvolutionEnhancementSettings()
        settings.lsmr.damping_factor = payload.damping_factor
        settings.lsmr.gradient_smoothness = payload.gradient_smoothness
        settings.lsmr.max_iter = payload.max_iterations
        settings.lsmr.atol = payload.atol
        settings.lsmr.btol = payload.btol
        settings.lsmr.checkpoint_interval = payload.checkpoint_interval
        settings.gpu.enabled = payload.use_gpu
        settings.gpu.index = payload.gpu_device_index
        # Fourier interpolation requires the GPU; fall back to Barycentric on CPU.
        settings._interpolation = (
            InterpolationTypes.FOURIER if payload.use_gpu else InterpolationTypes.BARYCENTRIC
        )

        algorithm = VSPIFluorescenceEnhancingAlgorithm()

        for enhanced_dataset, iteration in algorithm.enhance(dataset, product, settings=settings):
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
    except Exception:
        import traceback

        result_queue.put(('error', traceback.format_exc()))
    finally:
        ptychozoon_logger.removeHandler(log_handler)
        result_queue.put(None)

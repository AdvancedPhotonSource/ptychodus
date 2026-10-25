"""Child-side subprocess entry point for the Kickstart reconstructor.

One pass accumulates the moments of the input's patterns, keyed by the indexes of the
positions they are aligned with, and the estimate is made from them alone.
"""

from __future__ import annotations

from multiprocessing.queues import Queue
import pickle
from typing import Any

from ptychodus.api.accumulate import compute_diffraction_moments
from ptychodus.api.kickstart import estimate_product_from_moments
from ptychodus.api.reconstruct import ReconstructOutput, ReconstructorMessage, SubprocessPayload


def run_reconstruct(payload: SubprocessPayload, queue: Queue[Any]) -> None:
    """Child entry point for one estimate. Streams a single ReconstructOutput."""
    input_ = payload.require_input()
    product = input_.product
    object_in = product.object_
    moments = compute_diffraction_moments(
        # A reconstruct input holds one position per pattern, in pattern order.
        product.probe_positions.get_indexes(),
        input_.diffraction_patterns,
        bad_pixels=input_.bad_pixels,
        background=input_.background,
        flat_field=input_.flat_field,
        binning=payload.options['binning'],
    )

    if moments is None:
        raise ValueError('Kickstart has no diffraction patterns to estimate from.')

    estimate = estimate_product_from_moments(
        moments,
        product.probe_positions,
        metadata=product.metadata,
        probe_geometry=product.probes.get_geometry(),
        object_geometry=object_in.get_geometry(),
        layer_spacing_m=object_in.layer_spacing_m,
        parameters=payload.options['parameters'],
    )
    queue.put(
        (
            ReconstructorMessage.OUTPUT,
            pickle.dumps(ReconstructOutput(product=estimate.product, progress=1, progress_goal=1)),
        )
    )

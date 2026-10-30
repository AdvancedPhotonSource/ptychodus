"""Child side of the ptychozoon enhancer, run in a freshly spawned process.

Every ptychozoon import is deferred into the entry point, so the CuPy context is
created cleanly in the child and fully released when it exits. The payload carries
ptychodus types; the ptychozoon objects are built here, which is what keeps the
parent from importing ptychozoon at all. Log forwarding and error marshaling are
handled by the shared subprocess protocol in :mod:`ptychodus.api.spawn`.
"""

from __future__ import annotations

from importlib import import_module
from multiprocessing.queues import SimpleQueue
from typing import Any
import logging

import numpy

from ptychodus.api.fluorescence import (
    ElementMap,
    FluorescenceDataset,
    FluorescenceEnhancerOutput,
)

from ._payload import PtychozoonMessage, PtychozoonPayload

__all__ = [
    'run_enhancement',
]

logger = logging.getLogger(__name__)


def _count_visible_gpus() -> int:
    """How many CUDA devices this child can see once the device mask is applied.

    Asked here rather than in the parent, because asking CuPy acquires the context
    the parent must never hold. A mask that hides every device makes the runtime
    raise rather than report zero.
    """
    # Through importlib because CuPy ships no type information, and naming it in the
    # project's list of untyped modules would excuse it everywhere rather than here.
    cupy = import_module('cupy')

    try:
        return int(cupy.cuda.runtime.getDeviceCount())
    except cupy.cuda.runtime.CUDARuntimeError:
        return 0


def run_enhancement(payload: PtychozoonPayload, result_queue: SimpleQueue[Any]) -> None:
    """Run the ptychozoon enhancement and stream each checkpoint over *result_queue*.

    Intended as the ``entry_point`` for :func:`ptychodus.api.spawn.run_subprocess`.
    Puts one ``PtychozoonMessage.OUTPUT`` per checkpoint, carrying a
    :class:`FluorescenceEnhancerOutput` with the maps enhanced so far.

    Raises:
        RuntimeError: if the device mask leaves no CUDA device visible.
    """
    # Checked before ptychozoon is imported, because importing it is what fails:
    # ptychozoon times itself with chronos, which counts the GPUs at import and
    # raises when there are none. Its own CPU path is therefore out of reach.
    if _count_visible_gpus() == 0:
        raise RuntimeError(
            'ptychozoon needs a visible CUDA device, and the device selection leaves '
            'none. Select a GPU, or enhance on the CPU with the vspi enhancer.'
        )

    from ptychozoon.data_structures import ElementMap as PtychozoonElementMap
    from ptychozoon.data_structures import FluorescenceDataset as PtychozoonFluorescenceDataset
    from ptychozoon.data_structures import PtychographyProduct
    from ptychozoon.settings import DeconvolutionEnhancementSettings, InterpolationTypes
    from ptychozoon.vspi_enhance import VSPIFluorescenceEnhancingAlgorithm

    product = payload.input_.product
    dataset = payload.input_.dataset
    options = payload.options
    object_geometry = product.object_.get_geometry()

    probe_positions_m = numpy.array(
        [[p.y_m, p.x_m] for p in product.probe_positions],
        dtype=float,
    ).reshape((-1, 2))

    opr_weights = product.probes.get_opr_weights_or_none()
    opr_mode_weights = None if opr_weights is None else numpy.ascontiguousarray(opr_weights.T)

    ptychozoon_product = PtychographyProduct(
        probe_positions=probe_positions_m,
        probe=product.probes.get_array(),
        object_array=product.object_.get_layers_flattened(),
        pixel_size_m=(object_geometry.pixel_height_m, object_geometry.pixel_width_m),
        object_center_m=(object_geometry.center_y_m, object_geometry.center_x_m),
        opr_mode_weights=opr_mode_weights,
    )
    ptychozoon_dataset = PtychozoonFluorescenceDataset(
        element_maps=[
            PtychozoonElementMap(name=emap.name, counts_per_second=emap.counts_per_second)
            for emap in dataset
        ]
    )

    settings = DeconvolutionEnhancementSettings()
    settings.lsmr.damping_factor = options.damping_factor
    settings.lsmr.max_iter = options.max_iter
    settings.lsmr.atol = options.atol
    settings.lsmr.btol = options.btol
    settings.lsmr.checkpoint_interval = options.checkpoint_interval

    # Placement was decided by the parent and applied as a device mask, so device 0
    # is the first device the mask admits.
    settings.gpu.enabled = True
    settings.gpu.index = 0
    settings._interpolation = InterpolationTypes.FOURIER

    algorithm = VSPIFluorescenceEnhancingAlgorithm()

    for enhanced_dataset, iteration in algorithm.enhance(
        ptychozoon_dataset, ptychozoon_product, settings=settings
    ):
        result_queue.put(
            (
                PtychozoonMessage.OUTPUT,
                FluorescenceEnhancerOutput(
                    dataset=FluorescenceDataset(
                        _element_maps=[
                            ElementMap(emap.name, numpy.asarray(emap.counts_per_second))
                            for emap in enhanced_dataset.element_maps
                        ],
                        counts_per_second_path=dataset.counts_per_second_path,
                        channel_names_path=dataset.channel_names_path,
                    ),
                    progress=int(iteration),
                    progress_goal=options.max_iter,
                ),
            )
        )

"""GPU-accelerated VSPI fluorescence enhancer backed by the ptychozoon package.

The heavy CuPy computation runs in a freshly ``spawn``ed subprocess (see
:mod:`._ptychozoon_subprocess`) so each run gets a clean GPU context and all GPU
memory is released when the run finishes or is stopped. This module never imports
ptychozoon or CuPy directly.
"""

from __future__ import annotations
from collections.abc import Iterator
from typing import Final
import logging
import multiprocessing

import numpy

from ptychodus.api.fluorescence import (
    ElementMap,
    FluorescenceDataset,
    FluorescenceEnhancer,
    FluorescenceEnhancerInput,
    FluorescenceEnhancerOutput,
)
from ptychodus.api.observer import Observable, Observer
from ptychodus.api.product import Product

from ._ptychozoon_subprocess import PtychozoonPayload, run_vspi_enhancement
from .settings import FluorescenceSettings

logger = logging.getLogger(__name__)

__all__ = [
    'PtychozoonFluorescenceEnhancer',
]


class PtychozoonFluorescenceEnhancer(FluorescenceEnhancer, Observable, Observer):
    SIMPLE_NAME: Final[str] = 'VSPI-GPU'
    DISPLAY_NAME: Final[str] = 'Virtual Single Pixel Imaging (GPU)'

    def __init__(self, settings: FluorescenceSettings) -> None:
        super().__init__()
        self._settings = settings

        settings.ptychozoon_damping_factor.add_observer(self)
        settings.ptychozoon_gradient_smoothness.add_observer(self)
        settings.ptychozoon_max_iterations.add_observer(self)
        settings.ptychozoon_atol.add_observer(self)
        settings.ptychozoon_btol.add_observer(self)
        settings.ptychozoon_checkpoint_interval.add_observer(self)
        settings.ptychozoon_use_gpu.add_observer(self)
        settings.ptychozoon_gpu_device_index.add_observer(self)

    @property
    def name(self) -> str:
        return self.DISPLAY_NAME

    def get_progress_goal(self) -> int:
        return self._settings.ptychozoon_max_iterations.get_value()

    def _build_payload(self, parameters: FluorescenceEnhancerInput) -> PtychozoonPayload:
        product: Product = parameters.product
        dataset = parameters.dataset
        object_geometry = product.object_.get_geometry()

        probe_positions_m = numpy.array(
            [[p.coordinate_y_m, p.coordinate_x_m] for p in product.probe_positions],
            dtype=float,
        ).reshape((-1, 2))

        opr_weights = product.probes.get_opr_weights_or_none()
        opr_mode_weights = None if opr_weights is None else numpy.ascontiguousarray(opr_weights.T)

        return PtychozoonPayload(
            probe_positions_m=probe_positions_m,
            probe=product.probes.get_array(),
            object_array=product.object_.get_layers_flattened(),
            pixel_size_m=(object_geometry.pixel_height_m, object_geometry.pixel_width_m),
            object_center_m=(object_geometry.center_y_m, object_geometry.center_x_m),
            opr_mode_weights=opr_mode_weights,
            element_maps=[
                (emap.name, numpy.asarray(emap.counts_per_second)) for emap in dataset.element_maps
            ],
            damping_factor=self._settings.ptychozoon_damping_factor.get_value(),
            gradient_smoothness=self._settings.ptychozoon_gradient_smoothness.get_value(),
            max_iterations=self._settings.ptychozoon_max_iterations.get_value(),
            atol=self._settings.ptychozoon_atol.get_value(),
            btol=self._settings.ptychozoon_btol.get_value(),
            checkpoint_interval=self._settings.ptychozoon_checkpoint_interval.get_value(),
            use_gpu=self._settings.ptychozoon_use_gpu.get_value(),
            gpu_device_index=self._settings.ptychozoon_gpu_device_index.get_value(),
            log_level=logger.getEffectiveLevel(),
        )

    def enhance(
        self, parameters: FluorescenceEnhancerInput
    ) -> Iterator[FluorescenceEnhancerOutput]:
        dataset = parameters.dataset
        payload = self._build_payload(parameters)

        # A fresh spawned process gives ptychozoon/CuPy a clean GPU context and
        # releases all GPU memory when it exits.
        ctx = multiprocessing.get_context('spawn')
        result_queue = ctx.Queue()
        process = ctx.Process(target=run_vspi_enhancement, args=(payload, result_queue))
        process.start()

        try:
            while True:
                item = result_queue.get()

                if item is None:
                    break

                tag = item[0]

                if tag == 'log':
                    _, levelno, message = item
                    # Re-emit through this process's logger so it reaches the
                    # fluorescence status view via the registered handler.
                    logger.log(levelno, message)
                    continue

                if tag == 'error':
                    raise RuntimeError(f'ptychozoon enhancement failed:\n{item[1]}')

                if tag == 'result':
                    _, iteration, enhanced_maps = item
                    element_maps = [ElementMap(name, cps) for name, cps in enhanced_maps]
                    yield FluorescenceEnhancerOutput(
                        dataset=FluorescenceDataset(
                            element_maps=element_maps,
                            counts_per_second_path=dataset.counts_per_second_path,
                            channel_names_path=dataset.channel_names_path,
                        ),
                        progress=iteration,
                    )
                    continue
        finally:
            if process.is_alive():
                process.terminate()

            process.join(timeout=10.0)

            if process.is_alive():
                process.kill()
                process.join()

    def get_damping_factor(self) -> float:
        return self._settings.ptychozoon_damping_factor.get_value()

    def set_damping_factor(self, factor: float) -> None:
        self._settings.ptychozoon_damping_factor.set_value(factor)

    def get_gradient_smoothness(self) -> float:
        return self._settings.ptychozoon_gradient_smoothness.get_value()

    def set_gradient_smoothness(self, value: float) -> None:
        self._settings.ptychozoon_gradient_smoothness.set_value(value)

    def get_max_iterations(self) -> int:
        return self._settings.ptychozoon_max_iterations.get_value()

    def set_max_iterations(self, number: int) -> None:
        self._settings.ptychozoon_max_iterations.set_value(number)

    def get_atol(self) -> float:
        return self._settings.ptychozoon_atol.get_value()

    def set_atol(self, value: float) -> None:
        self._settings.ptychozoon_atol.set_value(value)

    def get_btol(self) -> float:
        return self._settings.ptychozoon_btol.get_value()

    def set_btol(self, value: float) -> None:
        self._settings.ptychozoon_btol.set_value(value)

    def get_checkpoint_interval(self) -> int:
        return self._settings.ptychozoon_checkpoint_interval.get_value()

    def set_checkpoint_interval(self, number: int) -> None:
        self._settings.ptychozoon_checkpoint_interval.set_value(number)

    def is_gpu_enabled(self) -> bool:
        return self._settings.ptychozoon_use_gpu.get_value()

    def set_gpu_enabled(self, enabled: bool) -> None:
        self._settings.ptychozoon_use_gpu.set_value(enabled)

    def get_gpu_device_index(self) -> int:
        return self._settings.ptychozoon_gpu_device_index.get_value()

    def set_gpu_device_index(self, index: int) -> None:
        self._settings.ptychozoon_gpu_device_index.set_value(index)

    def _update(self, observable: Observable) -> None:
        if observable in (
            self._settings.ptychozoon_damping_factor,
            self._settings.ptychozoon_gradient_smoothness,
            self._settings.ptychozoon_max_iterations,
            self._settings.ptychozoon_atol,
            self._settings.ptychozoon_btol,
            self._settings.ptychozoon_checkpoint_interval,
            self._settings.ptychozoon_use_gpu,
            self._settings.ptychozoon_gpu_device_index,
        ):
            self.notify_observers()

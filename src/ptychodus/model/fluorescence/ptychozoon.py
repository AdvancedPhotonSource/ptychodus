"""GPU-accelerated VSPI fluorescence enhancer backed by the ptychozoon package.

The heavy CuPy computation runs in a freshly ``spawn``ed subprocess (see
:mod:`._subprocess`) so each run gets a clean GPU context and all GPU memory
is released when the run finishes or is stopped. This module only imports the
CPU-safe ``ptychozoon.data_structures`` and ``ptychozoon.settings`` submodules
(needed to construct the payload); CuPy-linked submodules stay inside the
child.
"""

from __future__ import annotations
from collections.abc import Iterator
from typing import Final
import logging

import numpy

from ptychodus.api.fluorescence import (
    ElementMap,
    FluorescenceDataset,
    FluorescenceEnhancer,
    FluorescenceEnhancerInput,
    FluorescenceEnhancerOutput,
)
from ptychodus.api.product import Product

from ..processing._subprocess_protocol import run_subprocess
from ._payload import PtychozoonPayload
from .settings import FluorescenceSettings

_ENTRY_POINT: Final[str] = 'ptychodus.model.fluorescence._subprocess:run_vspi_enhancement'

logger = logging.getLogger(__name__)

__all__ = [
    'PtychozoonFluorescenceEnhancer',
]


class PtychozoonFluorescenceEnhancer(FluorescenceEnhancer):
    SIMPLE_NAME: Final[str] = 'VSPI-GPU'
    DISPLAY_NAME: Final[str] = 'Virtual Single Pixel Imaging (GPU)'

    def __init__(self, settings: FluorescenceSettings) -> None:
        super().__init__()
        self._settings = settings

    @property
    def name(self) -> str:
        return self.DISPLAY_NAME

    def get_progress_goal(self) -> int:
        return self._settings.ptychozoon_max_iterations.get_value()

    def _build_payload(self, parameters: FluorescenceEnhancerInput) -> PtychozoonPayload:
        # Imported here, not at module scope, so this module stays importable without
        # the optional ptychozoon extra. FluorescenceCore gates registration on
        # availability, but the class itself must always import -- the GUI enhance
        # dialog reads DISPLAY_NAME off it. Only the two CPU-safe submodules are
        # touched; see the allow-list in tests/test_no_gpu_context.py.
        from ptychozoon.data_structures import ElementMap as PtychozoonElementMap
        from ptychozoon.data_structures import FluorescenceDataset as PtychozoonFluorescenceDataset
        from ptychozoon.data_structures import PtychographyProduct
        from ptychozoon.settings import DeconvolutionEnhancementSettings, InterpolationTypes

        product: Product = parameters.product
        dataset = parameters.dataset
        object_geometry = product.object_.get_geometry()

        probe_positions_m = numpy.array(
            [[p.coordinate_y_m, p.coordinate_x_m] for p in product.probe_positions],
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
        settings.lsmr.damping_factor = self._settings.ptychozoon_damping_factor.get_value()
        settings.lsmr.gradient_smoothness = (
            self._settings.ptychozoon_gradient_smoothness.get_value()
        )
        settings.lsmr.max_iter = self._settings.ptychozoon_max_iterations.get_value()
        settings.lsmr.atol = self._settings.ptychozoon_atol.get_value()
        settings.lsmr.btol = self._settings.ptychozoon_btol.get_value()
        settings.lsmr.checkpoint_interval = (
            self._settings.ptychozoon_checkpoint_interval.get_value()
        )
        use_gpu = self._settings.ptychozoon_use_gpu.get_value()
        settings.gpu.enabled = use_gpu
        settings.gpu.index = self._settings.ptychozoon_gpu_device_index.get_value()
        # Fourier interpolation requires the GPU; fall back to Barycentric on CPU.
        settings._interpolation = (
            InterpolationTypes.FOURIER if use_gpu else InterpolationTypes.BARYCENTRIC
        )

        return PtychozoonPayload(
            product=ptychozoon_product,
            dataset=ptychozoon_dataset,
            settings=settings,
        )

    def enhance(
        self, parameters: FluorescenceEnhancerInput
    ) -> Iterator[FluorescenceEnhancerOutput]:
        dataset = parameters.dataset
        payload = self._build_payload(parameters)

        # A fresh spawned process gives ptychozoon/CuPy a clean GPU context and
        # releases all GPU memory when it exits. Log forwarding and error
        # marshaling live in the shared subprocess protocol.
        with run_subprocess(_ENTRY_POINT, payload) as events:
            for event in events:
                if event[0] != 'result':
                    continue
                _, iteration, enhanced_maps = event
                element_maps = [ElementMap(name, cps) for name, cps in enhanced_maps]
                yield FluorescenceEnhancerOutput(
                    dataset=FluorescenceDataset(
                        _element_maps=element_maps,
                        counts_per_second_path=dataset.counts_per_second_path,
                        channel_names_path=dataset.channel_names_path,
                    ),
                    progress=iteration,
                )

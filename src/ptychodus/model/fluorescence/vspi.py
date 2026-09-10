from __future__ import annotations
from collections.abc import Iterator
from typing import Final
import logging
import time

from scipy.sparse.linalg import lsmr, LinearOperator
import numpy

from ptychodus.api.typing import RealArrayType
from ptychodus.api.fluorescence import (
    ElementMap,
    FluorescenceDataset,
    FluorescenceEnhancer,
    FluorescenceEnhancerInput,
    FluorescenceEnhancerOutput,
)
from ptychodus.api.interpolate import BarycentricArrayInterpolator
from ptychodus.api.product import Product

from .settings import FluorescenceSettings

logger = logging.getLogger(__name__)

__all__ = [
    'VSPIFluorescenceEnhancer',
]


class VSPILinearOperator(LinearOperator):
    def __init__(self, product: Product) -> None:
        """
        M: number of XRF positions
        N: number of ptychography object pixels
        P: number of XRF channels

        A[M,N] * X[N,P] = B[M,P]
        """
        object_geometry = product.object_.get_geometry()
        M = len(product.probe_positions)  # noqa: N806
        N = object_geometry.height_px * object_geometry.width_px  # noqa: N806
        super().__init__(float, (M, N))
        self._product = product

    def _matvec(self, x: RealArrayType) -> RealArrayType:  # noqa: N803
        object_geometry = self._product.object_.get_geometry()
        object_array = x.reshape((object_geometry.height_px, object_geometry.width_px))
        AX = numpy.zeros(len(self._product.probe_positions))  # noqa: N806
        interpolator = BarycentricArrayInterpolator(object_array)

        for index, (scan_point, probe) in enumerate(self._product.iter_position_probes()):
            object_point = object_geometry.map_coordinates_probe_to_object(scan_point)
            probe_intensity = probe.get_intensity()
            psf = probe_intensity / probe_intensity.sum()
            patch = interpolator.get_patch(
                object_point.x_px,
                object_point.y_px,
                psf.shape[-1],
                psf.shape[-2],
            )
            AX[index] = numpy.sum(psf * patch)

        return AX

    def _rmatvec(self, x: RealArrayType) -> RealArrayType:  # noqa: N803
        object_geometry = self._product.object_.get_geometry()
        object_array = numpy.zeros((object_geometry.height_px, object_geometry.width_px))
        interpolator = BarycentricArrayInterpolator(object_array)

        for index, (scan_point, probe) in enumerate(self._product.iter_position_probes()):
            object_point = object_geometry.map_coordinates_probe_to_object(scan_point)
            probe_intensity = probe.get_intensity()
            psf = probe_intensity / probe_intensity.sum()
            interpolator.add_patch(
                object_point.x_px,
                object_point.y_px,
                x[index] * psf,
            )

        HX = object_array.flatten()  # noqa: N806

        return HX


class VSPIFluorescenceEnhancer(FluorescenceEnhancer):
    SIMPLE_NAME: Final[str] = 'VSPI'
    DISPLAY_NAME: Final[str] = 'Virtual Single Pixel Imaging'

    def __init__(self, settings: FluorescenceSettings) -> None:
        super().__init__()
        self._settings = settings

    @property
    def name(self) -> str:
        return self.DISPLAY_NAME

    def get_progress_goal(self) -> int:
        return 0

    def enhance(
        self, parameters: FluorescenceEnhancerInput
    ) -> Iterator[FluorescenceEnhancerOutput]:
        # Per-channel granularity; per-LSMR-iteration progress is a future enhancement
        # (would require wrapping LinearOperator._matvec or chunking lsmr with maxiter).
        dataset = parameters.dataset
        product = parameters.product
        object_geometry = product.object_.get_geometry()
        e_cps_shape = object_geometry.height_px, object_geometry.width_px
        element_maps: list[ElementMap] = list()
        A = VSPILinearOperator(product)  # noqa: N806

        for emap in dataset:
            logger.info(f'Enhancing "{emap.name}"...')
            tic = time.perf_counter()
            m_cps = emap.counts_per_second
            result = lsmr(
                A,
                m_cps.flatten(),
                damp=self._settings.vspi_damping_factor.get_value(),
                maxiter=self._settings.vspi_max_iterations.get_value(),
                show=True,
            )
            logger.debug(result)
            e_cps = result[0].reshape(e_cps_shape)
            emap_enhanced = ElementMap(emap.name, e_cps)
            toc = time.perf_counter()
            logger.info(f'Enhanced "{emap.name}" in {toc - tic:.4f} seconds.')

            element_maps.append(emap_enhanced)
            yield FluorescenceEnhancerOutput(
                dataset=FluorescenceDataset(
                    _element_maps=list(element_maps),
                    counts_per_second_path=dataset.counts_per_second_path,
                    channel_names_path=dataset.channel_names_path,
                ),
                progress=len(element_maps),
            )

"""Virtual single pixel imaging: one least-squares solve per element map."""

from __future__ import annotations
from collections.abc import Iterator
import logging

from pydantic import BaseModel, Field
from scipy.sparse.linalg import lsmr

from ptychodus.api.fluorescence import (
    ElementMap,
    FluorescenceDataset,
    FluorescenceEnhancer,
    FluorescenceEnhancerInput,
    FluorescenceEnhancerOutput,
    VSPILinearOperator,
)

from ._options import expect_options

logger = logging.getLogger(__name__)

__all__ = [
    'VSPIFluorescenceEnhancer',
    'VSPIOptions',
]


class VSPIOptions(BaseModel):
    damping_factor: float = Field(
        0.0,
        ge=0.0,
        title='Damping Factor:',
        description='Tikhonov regularization of the least-squares solve; zero applies none.',
    )
    max_iterations: int = Field(
        100,
        ge=1,
        title='Max Iterations:',
        description='Upper bound on LSMR iterations per element map.',
    )


class VSPIFluorescenceEnhancer(FluorescenceEnhancer):
    """Each element map solved for on the object grid through the probe-weighted forward model."""

    def get_default_options(self) -> BaseModel:
        return VSPIOptions()

    def enhance(
        self, input_: FluorescenceEnhancerInput, *, options: BaseModel
    ) -> Iterator[FluorescenceEnhancerOutput]:
        vspi = expect_options(options, VSPIOptions)
        # Per-channel granularity; per-LSMR-iteration progress is a future enhancement
        # (would require wrapping LinearOperator._matvec or chunking lsmr with maxiter).
        dataset = input_.dataset
        product = input_.product
        object_geometry = product.object_.get_geometry()
        e_cps_shape = object_geometry.height_px, object_geometry.width_px
        element_maps: list[ElementMap] = list()
        A = VSPILinearOperator(product)  # noqa: N806

        for emap in dataset:
            logger.info(f'Enhancing "{emap.name}"...')
            m_cps = emap.counts_per_second
            result = lsmr(
                A,
                m_cps.flatten(),
                damp=vspi.damping_factor,
                maxiter=vspi.max_iterations,
                show=True,
            )
            logger.debug(result)
            e_cps = result[0].reshape(e_cps_shape)
            emap_enhanced = ElementMap(emap.name, e_cps)

            element_maps.append(emap_enhanced)
            yield FluorescenceEnhancerOutput(
                dataset=FluorescenceDataset(
                    _element_maps=list(element_maps),
                    counts_per_second_path=dataset.counts_per_second_path,
                    channel_names_path=dataset.channel_names_path,
                ),
                progress=len(element_maps),
                progress_goal=len(dataset),
            )

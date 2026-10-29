"""Upscale each element map onto the object grid, then deconvolve the probe from it."""

from __future__ import annotations
from collections.abc import Iterator
import logging

from pydantic import BaseModel, Field

from ptychodus.api.fluorescence import (
    ElementMap,
    FluorescenceDataset,
    FluorescenceEnhancer,
    FluorescenceEnhancerInput,
    FluorescenceEnhancerOutput,
)

from ._options import expect_options
from .deconvolution import DeconvolutionMethod, deconvolve
from .upscaling import UpscalingMethod, upscale

logger = logging.getLogger(__name__)

__all__ = [
    'TwoStepFluorescenceEnhancer',
    'TwoStepOptions',
]


class TwoStepOptions(BaseModel):
    upscaling_method: UpscalingMethod = Field(
        UpscalingMethod.LINEAR,
        title='Upscaling Method:',
        description='How values measured at the scan positions are spread onto the object grid.',
    )
    deconvolution_method: DeconvolutionMethod = Field(
        DeconvolutionMethod.RICHARDSON_LUCY,
        title='Deconvolution Method:',
        description='How the probe intensity is removed from the upscaled map.',
    )
    wiener_balance: float = Field(
        0.05,
        ge=0.0,
        title='Wiener Balance:',
        description='Regularization of the Wiener filter, trading noise against sharpness.',
    )


class TwoStepFluorescenceEnhancer(FluorescenceEnhancer):
    """Each element map upscaled onto the object grid, then deconvolved, one at a time."""

    def get_default_options(self) -> BaseModel:
        return TwoStepOptions()

    def enhance(
        self, input_: FluorescenceEnhancerInput, *, options: BaseModel
    ) -> Iterator[FluorescenceEnhancerOutput]:
        two_step = expect_options(options, TwoStepOptions)
        dataset = input_.dataset
        product = input_.product
        element_maps: list[ElementMap] = list()

        for emap in dataset:
            logger.info(f'Enhancing "{emap.name}"...')
            emap_upscaled = upscale(emap, product, two_step.upscaling_method)
            emap_enhanced = deconvolve(
                emap_upscaled,
                product,
                two_step.deconvolution_method,
                wiener_balance=two_step.wiener_balance,
            )

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

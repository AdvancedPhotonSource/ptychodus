"""Deconvolve the probe from an upscaled element map.

See https://scikit-image.org/docs/stable/api/skimage.restoration.html
"""

from __future__ import annotations
from enum import StrEnum

import skimage.restoration

from ptychodus.api.fluorescence import ElementMap
from ptychodus.api.product import Product

__all__ = [
    'DeconvolutionMethod',
    'deconvolve',
]


class DeconvolutionMethod(StrEnum):
    """How the probe's intensity is removed from an element map."""

    # TODO Implement method from https://doi.org/10.1364/OE.20.018287
    IDENTITY = 'identity'
    RICHARDSON_LUCY = 'richardson_lucy'
    WIENER = 'wiener'
    UNSUPERVISED_WIENER = 'unsupervised_wiener'


def deconvolve(
    emap: ElementMap,
    product: Product,
    method: DeconvolutionMethod,
    *,
    wiener_balance: float,
) -> ElementMap:
    """Return *emap* with the intensity of the probe of *product* deconvolved by *method*.

    *wiener_balance* is the Wiener filter's regularization, trading noise against
    sharpness; only :attr:`DeconvolutionMethod.WIENER` reads it.

    Raises:
        ValueError: for a method this function does not know.
    """
    if method is DeconvolutionMethod.IDENTITY:
        return emap

    probe_intensity = product.probes.get_probe_no_opr().get_intensity()

    match method:
        case DeconvolutionMethod.RICHARDSON_LUCY:
            cps = skimage.restoration.richardson_lucy(emap.counts_per_second, probe_intensity)
        case DeconvolutionMethod.WIENER:
            cps = skimage.restoration.wiener(
                emap.counts_per_second, probe_intensity, wiener_balance
            )
        case DeconvolutionMethod.UNSUPERVISED_WIENER:
            cps, _ = skimage.restoration.unsupervised_wiener(
                emap.counts_per_second, probe_intensity
            )
        case _:
            raise ValueError(f'Unknown deconvolution method "{method}"!')

    return ElementMap(emap.name, cps)

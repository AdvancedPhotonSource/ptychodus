"""XMCD (X-ray Magnetic Circular Dichroism) decomposition math.

The decomposition models the helicity-dependent complex transmission of a
magnetic sample as::

    O_+  (RCP)  =  O_struct  *  M
    O_-  (LCP)  =  O_struct  /  M

where ``O_struct`` is the helicity-independent (structural) transmission and
``M = |M| * exp(i * alpha)`` is the magnetic factor (carrying both absorption
and phase contributions). Given the two reconstructed objects ``O_+`` and
``O_-``, the magnetic factor is recovered first::

    magnetic_object    =  sqrt(|O_+ / O_-|) * exp(i * alpha),  alpha = 0.5 * angle(O_+ * conj(O_-))
    structural_object  =  sqrt(|O_+ * O_-|) * exp(i * angle(O_- * exp(i * alpha)))

so that ``structural * magnetic = O_+`` and ``structural / magnetic = O_-``
hold exactly.

Design notes:

1. The magnetic *phase* is taken from the cross product rather than from the
   ratio ``O_+ / O_-``, because the cross product is well-conditioned even at
   pixels where ``|O_-|`` is small (no division). The magnetic *amplitude*
   still requires the ratio, hence ``epsilon`` is added to the LCP denominator
   as a NaN guard at empty pixels.

2. The model determines ``(O_struct, M)`` only up to a joint sign: ``(-O_struct,
   -M)`` reproduces both measurements. Taking ``alpha`` on the principal branch
   ``(-pi/2, pi/2]`` picks ``M`` near the positive real axis, which is where a
   magnetic factor lies, and deriving the structural phase from ``alpha`` keeps
   the pair consistent. Halving the structural and magnetic angles on
   independent branches would instead flip the sign of ``O_struct`` alone
   wherever its phase crosses ``+-pi/2``. A magnetic phase outside the
   principal branch wraps by ``pi``, and the structural object flips sign with
   it.

3. Each reconstruction carries its own global scale, phase offset and phase
   ramp, so the LCP product is standardized against the RCP product before the
   decomposition. The fit cannot tell those ambiguities from the magnetic
   signal, so it also absorbs the weighted mean magnetic phase, its linear
   trend, and the weighted mean ``|M|**2``: magnetic contrast is relative to
   the field of view, and a uniform magnetization is not recovered.

4. Both results are products on the RCP product's frame: its probe, probe
   positions and metadata surround the decomposed objects. The magnetic
   product's object is the factor ``M`` rather than a transmission, so it is
   not a starting point for a reconstruction.
"""

from __future__ import annotations
from dataclasses import dataclass, replace
import logging
import time

import numpy

from .metrics import estimate_reconstruction_ambiguities
from .object import Object, RegistrationQuantity, align_objects
from .product import Product

__all__ = [
    'XMCDResult',
    'estimate_xmcd',
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class XMCDResult:
    """Structural / magnetic decomposition produced by :func:`estimate_xmcd`."""

    structural_product: Product
    magnetic_product: Product


def _decompose_objects(
    rcp_object: Object, lcp_object_aligned: Object, *, epsilon: float
) -> tuple[Object, Object]:
    """Decompose an aligned RCP/LCP object pair into ``(structural, magnetic)`` objects.

    Both inputs must share array shape and pixel geometry; multi-layer objects
    are flattened layer-wise. Both outputs are single-layer and stamped with
    ``rcp_object``'s pixel geometry and center: an aligned LCP object's center
    is offset by the alignment shift, but the math is elementwise on the arrays.
    """
    rcp_pixel_geometry = rcp_object.get_pixel_geometry()
    lcp_pixel_geometry = lcp_object_aligned.get_pixel_geometry()

    if rcp_pixel_geometry != lcp_pixel_geometry:
        raise ValueError(
            f'Object pixel geometry mismatch: rcp {rcp_pixel_geometry} vs lcp {lcp_pixel_geometry}!'
        )

    rcp_array = rcp_object.get_layers_flattened()
    lcp_array = lcp_object_aligned.get_layers_flattened()

    if rcp_array.shape != lcp_array.shape:
        raise ValueError(
            f'Object array shape mismatch: rcp {rcp_array.shape} vs lcp {lcp_array.shape}!'
        )

    cross_helicity_product = rcp_array * numpy.conj(lcp_array)
    parallel_helicity_ratio = rcp_array / (lcp_array + epsilon)

    magnetic_phasor = numpy.exp(0.5j * numpy.angle(cross_helicity_product))
    structural_absorption = numpy.sqrt(numpy.abs(cross_helicity_product))
    structural_phase = numpy.angle(lcp_array * magnetic_phasor)
    magnetic_absorption = numpy.sqrt(numpy.abs(parallel_helicity_ratio))

    rcp_center = rcp_object.get_center()
    structural_object = Object(
        array=structural_absorption * numpy.exp(1j * structural_phase),
        pixel_geometry=rcp_pixel_geometry,
        center=rcp_center,
        layer_spacing_m=[],
    )
    magnetic_object = Object(
        array=magnetic_absorption * magnetic_phasor,
        pixel_geometry=rcp_pixel_geometry,
        center=rcp_center,
        layer_spacing_m=[],
    )
    return structural_object, magnetic_object


def _derive_product(
    rcp_product: Product, object_: Object, *, component: str, lcp_name: str
) -> Product:
    rcp_name = rcp_product.metadata.name
    metadata = replace(
        rcp_product.metadata,
        name=f'{rcp_name}-{component}',
        comments=f"XMCD {component} component of LCP '{lcp_name}' and RCP '{rcp_name}'",
        polarization=None,
    )
    return replace(rcp_product, metadata=metadata, object_=object_, losses=[])


def estimate_xmcd(
    lcp_product: Product,
    rcp_product: Product,
    *,
    upsample_factor: int = 100,
    registration_quantity: RegistrationQuantity = RegistrationQuantity.COMPLEX,
    high_pass_sigma_px: float = 0.0,
    epsilon: float = 1.0e-12,
) -> XMCDResult:
    """Decompose an LCP/RCP product pair into structural and magnetic products.

    The LCP object is aligned to the RCP object (see
    :func:`ptychodus.api.object.align_objects`, which also trims both to their
    common shape), the aligned LCP product's scale, phase offset and phase ramp
    are standardized against the RCP product, and the pair is decomposed under
    the helicity model ``O_+ = O_struct * M``, ``O_- = O_struct / M`` (see the
    module docstring). Multi-layer objects are flattened layer-wise; per-layer
    decomposition is not implemented.

    Args:
        lcp_product: Left-circularly-polarized reconstruction (``O_-``).
        rcp_product: Right-circularly-polarized reconstruction (``O_+``), the
            reference frame of both results.
        upsample_factor: Sub-pixel precision of the alignment. See
            :func:`ptychodus.api.object.align_objects`.
        registration_quantity: Quantity correlated to align the objects. See
            :class:`ptychodus.api.object.RegistrationQuantity`.
        high_pass_sigma_px: Gaussian-background sigma in pixels removed before
            alignment, ``0.0`` to disable.
        epsilon: Small positive number added to the LCP array before dividing,
            to keep the magnetic amplitude finite at pixels where ``O_-`` is
            zero. Only protects against exact zeros; magnetic output at
            near-zero LCP pixels remains noisy by construction.

    Returns:
        :class:`XMCDResult` whose products are the RCP product with its object
        replaced by the structural or magnetic object, named
        ``<rcp name>-structural`` / ``<rcp name>-magnetic``, with provenance
        comments, no polarization and no losses.
    """
    if lcp_product.object_.num_layers > 1 or rcp_product.object_.num_layers > 1:
        logger.warning('XMCD flattens multi-layer objects; per-layer XMCD is not implemented.')

    logger.info('Computing object alignment...')
    tic = time.perf_counter()
    cropped_rcp_object, aligned_lcp_object = align_objects(
        rcp_product.object_,
        lcp_product.object_,
        upsample_factor=upsample_factor,
        registration_quantity=registration_quantity,
        high_pass_sigma_px=high_pass_sigma_px,
    )
    toc = time.perf_counter()
    logger.info(f'Computed object alignment in {toc - tic:.4f} seconds.')

    cropped_rcp_product = replace(rcp_product, object_=cropped_rcp_object)
    aligned_lcp_product = replace(lcp_product, object_=aligned_lcp_object)

    ambiguities = estimate_reconstruction_ambiguities(
        aligned_lcp_product, reference=cropped_rcp_product
    )
    standardized_lcp_product = ambiguities.standardize_product(aligned_lcp_product)

    logger.info('Computing XMCD...')
    tic = time.perf_counter()
    structural_object, magnetic_object = _decompose_objects(
        cropped_rcp_object, standardized_lcp_product.object_, epsilon=epsilon
    )
    toc = time.perf_counter()
    logger.info(f'Computed XMCD in {toc - tic:.4f} seconds.')

    lcp_name = lcp_product.metadata.name
    return XMCDResult(
        structural_product=_derive_product(
            rcp_product, structural_object, component='structural', lcp_name=lcp_name
        ),
        magnetic_product=_derive_product(
            rcp_product, magnetic_object, component='magnetic', lcp_name=lcp_name
        ),
    )

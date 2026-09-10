"""XMCD (X-ray Magnetic Circular Dichroism) decomposition math.

The decomposition models the helicity-dependent complex transmission of a
magnetic sample as::

    O_+  (RCP)  =  O_struct  *  M
    O_-  (LCP)  =  O_struct  /  M

where ``O_struct`` is the helicity-independent (structural) transmission and
``M = |M| * exp(i * alpha)`` is the magnetic factor (carrying both absorption
and phase contributions). Given the two reconstructed objects ``O_+`` and
``O_-``, three helicity products are formed::

    parallel_helicity_product  =  O_+ * O_-            =  O_struct**2
    parallel_helicity_ratio    =  O_+ / O_-            =  M**2
    cross_helicity_product     =  O_+ * conj(O_-)      =  |O_struct|**2 * exp(2i * alpha)

from which the structural and magnetic objects are recovered as::

    structural_object  =  sqrt(|parallel_helicity_product|) * exp(0.5i * angle(parallel_helicity_product))
    magnetic_object    =  sqrt(|parallel_helicity_ratio|)   * exp(0.5i * angle(cross_helicity_product))

Two design notes:

1. The magnetic *phase* is taken from the cross product rather than from the
   parallel ratio, because the cross product is well-conditioned even at
   pixels where ``|O_-|`` is small (no division). The magnetic *amplitude*
   still requires the ratio, hence ``epsilon`` is added to the LCP denominator
   as a NaN guard at empty pixels.

2. The half-angle extractions ``0.5 * angle(.)`` use the principal branch with
   the cut on the negative real axis. Structural and magnetic phases are
   therefore determined only modulo ``pi``; this is intrinsic to the XMCD
   decomposition and equivalent to the principal branch of ``sqrt`` on the
   underlying complex argument.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import numpy

from .object import Object

__all__ = [
    'XMCDResult',
    'estimate_xmcd',
]


@dataclass(frozen=True)
class XMCDResult:
    """Structural / magnetic decomposition produced by :func:`estimate_xmcd`."""

    structural_object: Object
    magnetic_object: Object

    def save_npz(self, file_path: Path) -> None:
        pixel_geometry = self.structural_object.get_pixel_geometry()
        center = self.structural_object.get_center()
        numpy.savez_compressed(
            file_path,
            allow_pickle=False,
            structural_object=self.structural_object.get_array(),
            magnetic_object=self.magnetic_object.get_array(),
            pixel_height_m=pixel_geometry.height_m,
            pixel_width_m=pixel_geometry.width_m,
            center_x_m=center.x_m,
            center_y_m=center.y_m,
        )


def estimate_xmcd(
    rcp_object: Object, lcp_object_aligned: Object, *, epsilon: float = 1.0e-12
) -> XMCDResult:
    """Decompose an aligned RCP/LCP object pair into structural and magnetic parts.

    Assumes the helicity model ``O_+ = O_struct * M`` and ``O_- = O_struct / M``
    (see module docstring). Both inputs must share array shape and pixel
    geometry; multi-layer objects are flattened layer-wise (per-layer
    decomposition is not implemented). Callers are responsible for warning users
    about the flattening when relevant. Shape parity is guaranteed by
    :func:`ptychodus.api.object.align_objects`, which trims mismatched RCP/LCP
    reconstructions to their common shape.

    Args:
        rcp_object: Right-circularly-polarized reconstruction (``O_+``). Should
            be the cropped RCP returned by
            :func:`ptychodus.api.object.align_objects` when the input pair had
            mismatched shapes.
        lcp_object_aligned: Left-circularly-polarized reconstruction (``O_-``),
            already spatially aligned to ``rcp_object`` (see
            :func:`ptychodus.api.object.align_objects`).
        epsilon: Small positive number added to the LCP array before dividing,
            to keep the magnetic amplitude finite at pixels where ``O_-`` is
            zero. Only protects against exact zeros; magnetic output at
            near-zero LCP pixels remains noisy by construction.

    Returns:
        :class:`XMCDResult` containing the structural and magnetic objects,
        both single-layer and stamped with ``rcp_object``'s pixel geometry and
        center.
    """
    rcp_center = rcp_object.get_center()
    # Aligned objects intentionally have offset centers encoding the alignment shift;
    # XMCD math is pure elementwise on the arrays, so the centers need not match.
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
    parallel_helicity_product = rcp_array * lcp_array
    parallel_helicity_ratio = rcp_array / (lcp_array + epsilon)

    structural_absorption = numpy.sqrt(numpy.abs(cross_helicity_product))
    structural_phase = 0.5 * numpy.angle(parallel_helicity_product)
    magnetic_absorption = numpy.sqrt(numpy.abs(parallel_helicity_ratio))
    magnetic_phase = 0.5 * numpy.angle(cross_helicity_product)

    return XMCDResult(
        structural_object=Object(
            array=structural_absorption * numpy.exp(1j * structural_phase),
            pixel_geometry=rcp_pixel_geometry,
            center=rcp_center,
            layer_spacing_m=[],
        ),
        magnetic_object=Object(
            array=magnetic_absorption * numpy.exp(1j * magnetic_phase),
            pixel_geometry=rcp_pixel_geometry,
            center=rcp_center,
            layer_spacing_m=[],
        ),
    )

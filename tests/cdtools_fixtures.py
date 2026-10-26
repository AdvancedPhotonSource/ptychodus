"""A small far-field product whose patterns already agree with it, for the cdtools tests.

The patterns are simulated with ptychodus's own placement, :func:`iter_probe_patches`,
and an orthonormal centered FFT, which is the forward model cdtools' FancyPtycho
implements. A cdtools model built from the product should therefore predict them, and
the probe gauge measured from it should be one.

Pixels are deliberately non-square, the object is off-center, and positions fall
between pixels, so a transposed axis, a dropped center or a half-pixel slip all show.
"""

from __future__ import annotations

from dataclasses import replace

import numpy

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.object import Object, ObjectCenter
from ptychodus.api.patch import iter_probe_patches
from ptychodus.api.probe import ProbeEnsemble
from ptychodus.api.probe_positions import ProbePosition, ProbePositionSequence
from ptychodus.api.product import Product, ProductMetadata
from ptychodus.api.reconstruct import ReconstructInput
from ptychodus.api.typing import ComplexArrayType, RealArrayType

PIXEL_WIDTH_M = 10.0e-9
PIXEL_HEIGHT_M = 12.0e-9
PROBE_HEIGHT_PX = 16
PROBE_WIDTH_PX = 20
OBJECT_HEIGHT_PX = 56
OBJECT_WIDTH_PX = 64
NUM_POSITIONS = 9
DETECTOR_DISTANCE_M = 2.0
PHOTON_ENERGY_EV = 10_000.0
OBJECT_CENTER = ObjectCenter(x_m=3.3e-9, y_m=-7.1e-9)
PIXEL_GEOMETRY = PixelGeometry(width_m=PIXEL_WIDTH_M, height_m=PIXEL_HEIGHT_M)

METADATA = ProductMetadata(
    name='cdtools',
    comments='',
    detector_distance_m=DETECTOR_DISTANCE_M,
    photon_energy_eV=PHOTON_ENERGY_EV,
    probe_photon_count=1.0e6,
    exposure_time_s=1.0,
    mass_attenuation_m2_per_kg=0.0,
    tomography_angle_deg=0.0,
)


def far_field_intensity(wavefield: ComplexArrayType) -> RealArrayType:
    """Detector intensity under a centered orthonormal FFT, cdtools' far field."""
    axes = (-2, -1)
    propagated = numpy.fft.fftshift(
        numpy.fft.fft2(numpy.fft.ifftshift(wavefield, axes=axes), norm='ortho'), axes=axes
    )
    return numpy.abs(propagated) ** 2


def make_probes(*, num_incoherent_modes: int = 2, scale: float = 1.0) -> ProbeEnsemble:
    yy, xx = numpy.mgrid[:PROBE_HEIGHT_PX, :PROBE_WIDTH_PX]
    cy = (PROBE_HEIGHT_PX - 1) / 2
    cx = (PROBE_WIDTH_PX - 1) / 2
    modes = [
        numpy.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 12.0 + 0.2j * xx),
        0.3 * numpy.exp(-((yy - cy + 1) ** 2 + (xx - cx - 2) ** 2) / 8.0),
    ][:num_incoherent_modes]
    array = (30.0 * scale * numpy.stack(modes)[numpy.newaxis]).astype(numpy.complex128)
    return ProbeEnsemble(array=array, opr_weights=None, pixel_geometry=PIXEL_GEOMETRY)


def make_object(*, num_layers: int = 1, seed: int = 1) -> Object:
    rng = numpy.random.default_rng(seed)
    shape = (num_layers, OBJECT_HEIGHT_PX, OBJECT_WIDTH_PX)
    array = rng.uniform(0.6, 1.0, shape) * numpy.exp(1j * rng.uniform(-1.0, 1.0, shape))
    return Object(
        array=array,
        pixel_geometry=PIXEL_GEOMETRY,
        center=OBJECT_CENTER,
        layer_spacing_m=[1.0e-6] * (num_layers - 1),
    )


def make_positions(*, seed: int = 2, spread_px: float = 10.0) -> ProbePositionSequence:
    rng = numpy.random.default_rng(seed)
    return ProbePositionSequence(
        [
            ProbePosition(
                index=10 + 3 * i,
                x_m=OBJECT_CENTER.x_m + float(rng.uniform(-spread_px, spread_px)) * PIXEL_WIDTH_M,
                y_m=OBJECT_CENTER.y_m + float(rng.uniform(-spread_px, spread_px)) * PIXEL_HEIGHT_M,
                probe_photon_count=1.0e6 + i,
            )
            for i in range(NUM_POSITIONS)
        ]
    )


def make_product(**kwargs: object) -> Product:
    """The fixture product; any keyword replaces that field of it."""
    product = Product(
        metadata=METADATA,
        probe_positions=make_positions(),
        probes=make_probes(),
        object_=make_object(),
        losses=[],
    )
    return replace(product, **kwargs)  # type: ignore[arg-type]


def simulate_patterns(product: Product) -> RealArrayType:
    """The patterns *product* predicts under cdtools' far field, one per position."""
    object_array = product.object_.get_layers_flattened()
    patterns = [
        numpy.sum(
            far_field_intensity(
                patch.modes * object_array[patch.bounds.y_slice, patch.bounds.x_slice]
            ),
            axis=0,
        )
        for patch in iter_probe_patches(product)
    ]
    return numpy.asarray(patterns)


def make_input(product: Product | None = None, **kwargs: object) -> ReconstructInput:
    """A reconstruct input whose patterns *product* predicts exactly.

    Keywords other than the product are passed to the input, so a test can attach a
    background or a flat field and fold it into the patterns itself.
    """
    product = make_product() if product is None else product
    patterns = simulate_patterns(product)
    return ReconstructInput(
        diffraction_patterns=patterns,
        bad_pixels=numpy.zeros(patterns.shape[-2:], dtype=numpy.bool_),
        product=product,
        **kwargs,  # type: ignore[arg-type]
    )

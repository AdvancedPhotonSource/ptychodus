"""Simulated far-field scans under a defocusable disk probe, shared by the kickstart tests.

The probe's far field is a uniform disk of radius `DISK_RADIUS_PX` detector pixels,
propagated from its focus by the defocus asked for; the sample is a smooth random phase
texture with weak absorption, or nothing at all. Patterns are simulated by the house
far-field forward model with Poisson noise, so the sign conventions every estimate is
checked against are that model's.
"""

from __future__ import annotations

from functools import cache

import numpy
from scipy.ndimage import gaussian_filter

from ptychodus.api.accumulate import DiffractionMoments, compute_diffraction_moments
from ptychodus.api.assemble import AssembledDiffractionData
from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.object import Object, ObjectCenter, ObjectGeometry
from ptychodus.api.preprocess.diffraction import BinningStep
from ptychodus.api.probe import Probe, ProbeEnsemble, ProbeGeometry
from ptychodus.api.probe_positions import ProbePosition, ProbePositionSequence
from ptychodus.api.product import Product, ProductMetadata
from ptychodus.api.simulate.diffraction import generate_diffraction_data
from ptychodus.api.simulate.probe import propagate_probe

PIXEL_SIZE_M = 50e-9
PROBE_PX = 64
OBJECT_PX = 160
DISK_RADIUS_PX = 8
PIXEL_GEOMETRY = PixelGeometry(width_m=PIXEL_SIZE_M, height_m=PIXEL_SIZE_M)
PROBE_GEOMETRY = ProbeGeometry(
    width_px=PROBE_PX,
    height_px=PROBE_PX,
    pixel_width_m=PIXEL_SIZE_M,
    pixel_height_m=PIXEL_SIZE_M,
)
OBJECT_GEOMETRY = ObjectGeometry(
    width_px=OBJECT_PX,
    height_px=OBJECT_PX,
    pixel_width_m=PIXEL_SIZE_M,
    pixel_height_m=PIXEL_SIZE_M,
    center_x_m=0.0,
    center_y_m=0.0,
)
METADATA = ProductMetadata(
    name='parallax',
    comments='',
    detector_distance_m=2.0,
    photon_energy_eV=10_000.0,
    probe_photon_count=1.0,
    exposure_time_s=1.0,
    mass_attenuation_m2_per_kg=0.0,
    tomography_angle_deg=0.0,
)
WAVELENGTH_M = METADATA.photon_wavelength_m
# lambda / NA**2 for the disk: about 1.3 mm, so 2 mm is past it and 0 well inside.
DEPTH_OF_FOCUS_M = 1.0 / (WAVELENGTH_M * (DISK_RADIUS_PX / (PROBE_PX * PIXEL_SIZE_M)) ** 2)


def make_probe(defocus_m: float) -> Probe:
    """The disk's focus, propagated *defocus_m* downstream."""
    q = numpy.arange(PROBE_PX) - PROBE_PX // 2
    disk = (numpy.hypot(q[:, numpy.newaxis], q) <= DISK_RADIUS_PX).astype(complex)
    focus = numpy.fft.fftshift(numpy.fft.ifft2(numpy.fft.ifftshift(disk)))
    probe = Probe(array=1000.0 * focus / numpy.abs(focus).max(), pixel_geometry=PIXEL_GEOMETRY)
    return propagate_probe(
        probe, photon_wavelength_m=WAVELENGTH_M, propagation_distance_m=defocus_m
    )


def make_transmission(*, featureless: bool) -> numpy.ndarray:
    if featureless:
        return numpy.ones((OBJECT_PX, OBJECT_PX), dtype=complex)

    texture = gaussian_filter(
        numpy.random.default_rng(3).standard_normal((OBJECT_PX, OBJECT_PX)), 4.0
    )
    return numpy.exp((-0.05 + 0.5j) * texture / texture.std())


def make_positions() -> list[ProbePosition]:
    """A jittered 19 x 19 raster at a 4-pixel step, indexed in scan order."""
    rng = numpy.random.default_rng(1)
    positions: list[ProbePosition] = list()

    for iy in range(-36, 37, 4):
        for ix in range(-36, 37, 4):
            positions.append(
                ProbePosition(
                    index=len(positions),
                    x_m=(ix + rng.uniform(-0.5, 0.5)) * PIXEL_SIZE_M,
                    y_m=(iy + rng.uniform(-0.5, 0.5)) * PIXEL_SIZE_M,
                )
            )

    return positions


@cache
def simulate_scan(defocus_m: float, *, featureless: bool = False) -> AssembledDiffractionData:
    probe = make_probe(defocus_m)
    product = Product(
        metadata=METADATA,
        probe_positions=ProbePositionSequence(make_positions()),
        probes=ProbeEnsemble(
            array=probe.get_array(), opr_weights=None, pixel_geometry=PIXEL_GEOMETRY
        ),
        object_=Object(
            array=make_transmission(featureless=featureless)[numpy.newaxis],
            pixel_geometry=PIXEL_GEOMETRY,
            center=ObjectCenter(x_m=0.0, y_m=0.0),
        ),
        losses=[],
    )
    return generate_diffraction_data(product, numpy.random.default_rng(0))


@cache
def accumulate_scan(
    defocus_m: float, *, featureless: bool = False, bin_size_px: int | None = 2
) -> DiffractionMoments:
    """The scan's moments, binned by *bin_size_px* square bins, or not at all for None."""
    data = simulate_scan(defocus_m, featureless=featureless)
    moments = compute_diffraction_moments(
        numpy.array(data.get_indexes()),
        numpy.array(data.get_patterns()),
        bad_pixels=numpy.array(data.get_bad_pixels()),
        binning=(
            None
            if bin_size_px is None
            else BinningStep(bin_size_x=bin_size_px, bin_size_y=bin_size_px)
        ),
    )
    assert moments is not None
    return moments

#!/usr/bin/env python
"""Check the Bragg/Laue exit-frame transform by simulating, reconstructing and comparing.

Off the forward direction the detector sees the sample projected along the exit beam,
and pty-chi reconstructs it unchanged once the probe positions are carried into that
exit frame by :meth:`AffineTransform.from_scattering_geometry`. This script tests that
claim end to end. It simulates a thin crystal surface with
:func:`generate_scattering_data` and writes two prepared directories holding the same
patterns:

- ``corrected``: positions carried into the exit frame;
- ``control``: positions left as the scan recorded them, the transmission assumption.

Both start from the same probe, the back-propagated mean pattern, and a uniform object.
Each is reconstructed with ``ptychodus-reconstruct`` and compared with the exit-frame
truth. At forward geometry the two runs are identical. Off it, the corrected run should
match the truth about as well as a forward run does, and the control should not.

Given ``--prepared-directory`` instead, it transforms an existing prepared directory --
from a reconstruction driver run with ``--prepare-only`` on real data -- into a new
one for ``ptychodus-reconstruct``, without reconstructing. The source is left untouched.

Example usage:

    python scripts/check_scattering_round_trip.py --geometry bragg-asymmetric \\
        --work-directory /tmp/bragg-asymmetric
    python scripts/check_scattering_round_trip.py --geometry laue \\
        --prepared-directory prepared --work-directory prepared_exit

Needs the ptychi extra for the simulated mode. No packaged dependency -- this file is
intentionally not shipped with the wheel.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import argparse
import logging
import shutil
import subprocess
import sys

import numpy

from ptychodus.api.affine import AffineTransform, transform_probe_positions
from ptychodus.api.assemble import AssembledDiffractionData
from ptychodus.api.constants import energy_eV_to_wavelength_m
from ptychodus.api.geometry import ImageExtent, PixelGeometry
from ptychodus.api.io import StandardFileLayout, load_product, save_diffraction_data, save_product
from ptychodus.api.metrics import (
    compute_root_mean_square_error,
    compute_structural_similarity,
    estimate_reconstruction_ambiguities,
)
from ptychodus.api.object import (
    Object,
    ObjectGeometry,
    center_crop_object,
    compute_object_geometry,
)
from ptychodus.api.probe import ProbeEnsemble, ProbeGeometry
from ptychodus.api.probe_positions import ProbePositionSequence
from ptychodus.api.product import Product, ProductMetadata
from ptychodus.api.scattering import ScatteringGeometry
from ptychodus.api.simulate.diffraction import generate_scattering_data
from ptychodus.api.simulate.object import (
    generate_gaussian_random_field_object,
    generate_uniform_object,
)
from ptychodus.api.simulate.probe import (
    apply_defocus_phase,
    generate_average_pattern_probe,
    generate_super_gaussian_probe,
    rescale_probe_intensity,
)
from ptychodus.api.simulate.probe_positions import generate_spiral_probe_positions

logger = logging.getLogger(Path(__file__).stem)

_GEOMETRY_PRESETS = {
    'forward': ScatteringGeometry(),
    # The detector arm slightly out of the horizontal plane, so the transform shears.
    'bragg-asymmetric': ScatteringGeometry(
        two_theta_deg=47.35, detector_gamma_deg=-0.27, incidence_angle_deg=24.15
    ),
    'bragg-symmetric': ScatteringGeometry(two_theta_deg=47.35, incidence_angle_deg=23.675),
    # The detector arm off axis with the sample at normal incidence.
    'laue': ScatteringGeometry(two_theta_deg=57.42, incidence_angle_deg=90.0),
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=Path(__file__).stem,
        description='Check the Bragg/Laue exit-frame transform by a simulated round trip.',
    )
    parser.add_argument(
        '--geometry', choices=sorted(_GEOMETRY_PRESETS), default='bragg-asymmetric', help='Preset.'
    )
    parser.add_argument('--two-theta-deg', type=float, help='Overrides the preset.')
    parser.add_argument('--detector-gamma-deg', type=float, help='Overrides the preset.')
    parser.add_argument('--incidence-angle-deg', type=float, help='Overrides the preset.')
    parser.add_argument(
        '--work-directory',
        type=Path,
        required=True,
        help='Where the prepared and reconstructed directories are written; with '
        '--prepared-directory, the transformed prepared directory itself.',
    )
    parser.add_argument(
        '--prepared-directory',
        type=Path,
        help='Transform this prepared directory instead of simulating one.',
    )
    parser.add_argument('--photon-energy-eV', type=float, default=11200.0)
    parser.add_argument('--detector-distance-m', type=float, default=0.89)
    parser.add_argument('--detector-pixel-size-m', type=float, default=55e-6)
    parser.add_argument('--detector-px', type=int, default=128)
    parser.add_argument('--num-positions', type=int, default=300)
    parser.add_argument(
        '--probe-photon-count', type=float, default=1e8, help='Photons per exposure.'
    )
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--ptychi-options', type=Path, help='Passed to ptychodus-reconstruct; stock LSQML if unset.'
    )
    parser.add_argument('--log-level', type=int, default=logging.INFO)
    return parser


def _resolve_geometry(args: argparse.Namespace) -> ScatteringGeometry:
    geometry = _GEOMETRY_PRESETS[args.geometry]

    for name in ('two_theta_deg', 'detector_gamma_deg', 'incidence_angle_deg'):
        value = getattr(args, name)

        if value is not None:
            geometry = replace(geometry, **{name: value})

    return geometry


def _describe(geometry: ScatteringGeometry, transform: AffineTransform) -> str:
    kind = 'forward' if geometry.is_forward else 'Bragg' if geometry.is_reflection else 'Laue'
    return (
        f'{kind} geometry {geometry}: positions map by '
        f'[[{transform.a00:+.6f}, {transform.a01:+.6f}], '
        f'[{transform.a10:+.6f}, {transform.a11:+.6f}]]'
    )


def _with_positions(product: Product, positions: ProbePositionSequence) -> Product:
    """*product* moved onto *positions*, with a uniform object on the canvas they need."""
    source = product.object_.get_array()
    geometry = compute_object_geometry(positions, product.probes.get_geometry())
    amplitude = float(numpy.mean(numpy.abs(source))) if source.size else 1.0
    return replace(
        product,
        probe_positions=positions,
        object_=generate_uniform_object(geometry, amplitude=amplitude),
        losses=[],
    )


def _write_prepared(directory: Path, data: AssembledDiffractionData, product: Product) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    save_diffraction_data(StandardFileLayout.DIFFRACTION.path(directory), data)
    save_product(StandardFileLayout.PRODUCT.path(directory), product)


def _transform_prepared(args: argparse.Namespace, geometry: ScatteringGeometry) -> None:
    source: Path = args.prepared_directory
    target: Path = args.work_directory

    if target.resolve() == source.resolve():
        raise SystemExit('--work-directory must differ from --prepared-directory.')

    transform = AffineTransform.from_scattering_geometry(geometry)
    logger.info(_describe(geometry, transform))
    product = load_product(StandardFileLayout.PRODUCT.path(source))
    positions = ProbePositionSequence(
        list(transform_probe_positions(product.probe_positions, transform))
    )
    target.mkdir(parents=True, exist_ok=True)

    for layout in (StandardFileLayout.DIFFRACTION, StandardFileLayout.PTYCHI_OPTIONS):
        if layout.path(source).is_file():
            shutil.copy2(layout.path(source), layout.path(target))

    save_product(StandardFileLayout.PRODUCT.path(target), _with_positions(product, positions))
    logger.info(f'Wrote {target}; reconstruct it with: ptychodus-reconstruct -i {target} -o OUT')


def _simulate(
    args: argparse.Namespace, geometry: ScatteringGeometry
) -> tuple[AssembledDiffractionData, Product, Product]:
    """The patterns, the exit-frame truth, and the lab-frame initial product."""
    rng = numpy.random.default_rng(args.seed)
    wavelength_m = energy_eV_to_wavelength_m(args.photon_energy_eV)
    probe_geometry = ProbeGeometry.from_far_field(
        PixelGeometry(width_m=args.detector_pixel_size_m, height_m=args.detector_pixel_size_m),
        ImageExtent(width_px=args.detector_px, height_px=args.detector_px),
        wavelength_m=wavelength_m,
        distance_m=args.detector_distance_m,
    )
    pixel_m = probe_geometry.pixel_width_m
    probe = generate_super_gaussian_probe(
        probe_geometry, annular_radius_m=0.0, fwhm_m=10.0 * pixel_m, order_parameter=2.0
    )
    probe = apply_defocus_phase(probe, photon_wavelength_m=wavelength_m, defocus_distance_m=2e-4)
    probe = rescale_probe_intensity(probe, args.probe_photon_count)
    positions = ProbePositionSequence(
        list(generate_spiral_probe_positions(args.num_positions, 1.5 * pixel_m))
    )

    # The scan frame's x is the surface's in-plane axis foreshortened by sin(incidence),
    # and its y is the surface's y, so this surface grid covers the whole lab canvas.
    lab_geometry = compute_object_geometry(positions, probe_geometry)
    incidence_sine = abs(numpy.sin(numpy.radians(geometry.incidence_angle_deg)))
    surface_pixel_m = pixel_m / 2.0
    surface_geometry = ObjectGeometry(
        width_px=int(numpy.ceil(1.5 * lab_geometry.width_m / incidence_sine / surface_pixel_m)),
        height_px=int(numpy.ceil(1.5 * lab_geometry.height_m / surface_pixel_m)),
        pixel_width_m=surface_pixel_m,
        pixel_height_m=surface_pixel_m,
        center_x_m=0.0,
        center_y_m=0.0,
    )
    field = generate_gaussian_random_field_object(
        rng, surface_geometry, correlation_length_px=16.0
    ).get_layer(0)
    field /= numpy.std(field)
    # Amplitude contrast, and a phase standing in for G · u.
    surface = numpy.clip(1.0 + 0.3 * field.real, 0.2, None) * numpy.exp(1j * field.imag)

    metadata = ProductMetadata(
        name='scattering-truth',
        comments=f'Simulated {geometry}',
        detector_distance_m=args.detector_distance_m,
        photon_energy_eV=args.photon_energy_eV,
        probe_photon_count=args.probe_photon_count,
        exposure_time_s=1.0,
        mass_attenuation_m2_per_kg=0.0,
        tomography_angle_deg=0.0,
    )
    lab_product = Product(
        metadata=metadata,
        probe_positions=positions,
        probes=ProbeEnsemble.from_probe(probe),
        object_=Object(
            array=surface,
            pixel_geometry=surface_geometry.get_pixel_geometry(),
            center=surface_geometry.get_center(),
        ),
        losses=[],
    )
    data, truth = generate_scattering_data(lab_product, geometry, rng)

    initial_probe = generate_average_pattern_probe(
        probe_geometry,
        data.get_mean_pattern(),
        bad_pixels=data.get_bad_pixels(),
        photon_wavelength_m=wavelength_m,
        detector_distance_m=args.detector_distance_m,
    )
    initial = replace(
        lab_product,
        metadata=replace(metadata, name='scattering-initial'),
        probes=ProbeEnsemble.from_probe(initial_probe),
        object_=generate_uniform_object(lab_geometry),
    )
    return data, truth, initial


def _score(truth: Product, result: Product) -> str:
    """Agreement with the truth over the middle half of the scan's exit-frame bounding box.

    Both objects are center-cropped to a common canvas. Simulated positions are exact, so
    nothing is registered: only the global phase, scale and phase ramp are removed, which
    no measurement fixes. A registration would also hide what the control gets wrong.
    """
    height_px = min(truth.object_.height_px, result.object_.height_px)
    width_px = min(truth.object_.width_px, result.object_.width_px)
    reference = replace(truth, object_=center_crop_object(truth.object_, height_px, width_px))
    test = replace(result, object_=center_crop_object(result.object_, height_px, width_px))

    geometry = reference.object_.get_geometry()
    x_px = geometry.map_probe_positions_to_object_x_px(truth.probe_positions)
    y_px = geometry.map_probe_positions_to_object_y_px(truth.probe_positions)
    rows = slice(int(numpy.percentile(y_px, 25)), int(numpy.ceil(numpy.percentile(y_px, 75))) + 1)
    columns = slice(
        int(numpy.percentile(x_px, 25)), int(numpy.ceil(numpy.percentile(x_px, 75))) + 1
    )
    region = numpy.zeros((height_px, width_px))
    region[rows, columns] = 1.0

    ambiguities = estimate_reconstruction_ambiguities(test, reference=reference, weights=region)
    expected = reference.object_.get_layer(0)[rows, columns]
    actual = ambiguities.standardize_product(test).object_.get_layer(0)[rows, columns]
    relative_rmse = compute_root_mean_square_error(expected, actual) / float(
        numpy.sqrt(numpy.mean(numpy.square(numpy.abs(expected))))
    )
    phase_ssim = compute_structural_similarity(numpy.angle(expected), numpy.angle(actual))
    loss = result.losses[-1].value if result.losses else float('nan')
    return f'relative RMSE {relative_rmse:.3f}, phase SSIM {phase_ssim:.3f}, final loss {loss:.4g}'


def _reconstruct(args: argparse.Namespace, prepared: Path, output: Path) -> Product:
    command = [
        sys.executable,
        '-m',
        'ptychodus.cli.reconstruct_prepared',
        '-i',
        str(prepared),
        '-o',
        str(output),
    ]

    if args.ptychi_options is not None:
        command += ['--ptychi-options', str(args.ptychi_options)]

    logger.info('Running %s', ' '.join(command))
    subprocess.run(command, check=True)
    return load_product(StandardFileLayout.PRODUCT.path(output))


def main() -> int:
    args = _build_parser().parse_args()
    logging.basicConfig(level=args.log_level, format='%(name)s %(levelname)s: %(message)s')
    geometry = _resolve_geometry(args)

    if args.prepared_directory is not None:
        _transform_prepared(args, geometry)
        return 0

    transform = AffineTransform.from_scattering_geometry(geometry)
    logger.info(_describe(geometry, transform))
    data, truth, initial = _simulate(args, geometry)
    work: Path = args.work_directory
    work.mkdir(parents=True, exist_ok=True)
    save_product(work / 'truth.h5', truth)
    corrected_positions = ProbePositionSequence(
        list(transform_probe_positions(initial.probe_positions, transform))
    )
    runs = {
        'corrected': _with_positions(initial, corrected_positions),
        'control': _with_positions(initial, initial.probe_positions),
    }
    scores = {}

    for name, product in runs.items():
        _write_prepared(work / f'{name}_prepared', data, product)
        result = _reconstruct(args, work / f'{name}_prepared', work / f'{name}_result')
        scores[name] = _score(truth, result)

    print(_describe(geometry, transform))

    for name, score in scores.items():
        print(f'  {name:>9}: {score}')

    return 0


if __name__ == '__main__':
    sys.exit(main())

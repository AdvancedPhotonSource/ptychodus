#!/usr/bin/env python
"""Check a detector's orientation relative to the scan from the bright-field parallax.

A defocused probe makes each bright-field pixel see the sample shifted in proportion to
its spatial frequency, ``s = a + M q``. Fitting the matrix ``M`` over the bright field
names the flips and transpose that bring the detector's frames into the scan's frame,
and what is left: a scan rotation no quarter turn expresses, or a miscalibrated position
axis. A defocus sign error and a half-turn of the detector look the same to the
parallax, so both readings are reported; knowing the sign of the defocus picks one.

The probe must be defocused beyond its depth of focus and the sample textured, and the
patterns are read as the instrument recorded them, before any orientation steps.

Example usage:

    python scripts/check_detector_orientation.py --list-types
    python scripts/check_detector_orientation.py \\
        --patterns-type APS2IDDiffraction --patterns scan_0042_master.h5 \\
        --positions-type MDA --positions 2idd_0042.mda --figure orientation.png

No packaged dependency -- this file is intentionally not shipped with the wheel.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import logging
import math
import sys
from typing import Any, TypeVar

import numpy
from scipy.spatial import cKDTree

from ptychodus.api.accumulate import DiffractionMomentAccumulator, DiffractionMoments
from ptychodus.api.constants import energy_eV_to_wavelength_m
from ptychodus.api.diffraction import DiffractionDataset
from ptychodus.api.geometry import ImageExtent, PixelGeometry
from ptychodus.api.object import ObjectGeometry
from ptychodus.api.parallax import (
    OrientationCandidate,
    ParallaxTransform,
    estimate_parallax_transform,
)
from ptychodus.api.plugins import PluginChooser, PluginRegistry
from ptychodus.api.preprocess.diffraction import BinningStep
from ptychodus.api.probe import ProbeGeometry
from ptychodus.api.probe_positions import ProbePositionSequence, calculate_scan_geometry

logger = logging.getLogger(Path(__file__).stem)

_ReaderT = TypeVar('_ReaderT')


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=Path(__file__).stem,
        description='Check the detector orientation relative to the scan from the '
        'bright-field parallax of a defocused probe.',
    )
    parser.add_argument('--list-types', action='store_true', help='List the reader names.')
    parser.add_argument('--patterns-type', help='Diffraction file reader name.')
    parser.add_argument('--patterns', type=Path, help='Diffraction file.')
    parser.add_argument('--positions-type', help='Probe position file reader name.')
    parser.add_argument('--positions', type=Path, help='Probe position file.')
    parser.add_argument('--energy-eV', type=float, help='Photon energy; overrides the file.')
    parser.add_argument(
        '--detector-distance-m', type=float, help='Sample to detector; overrides the file.'
    )
    parser.add_argument(
        '--pixel-size-m', type=float, help='Detector pixel width; overrides the file.'
    )
    parser.add_argument(
        '--pixel-height-m', type=float, help='Detector pixel height, if not square.'
    )
    parser.add_argument(
        '--bin-size-px', type=int, default=4, help='Detector pixels per bin side (default 4).'
    )
    parser.add_argument(
        '--bright-field-fraction',
        type=float,
        default=0.5,
        help='Fraction of the brightest bin a bright-field bin reaches (default 0.5).',
    )
    parser.add_argument(
        '--image-pixel-m',
        type=float,
        help='Pitch of the bright-field images; default half the median scan step.',
    )
    parser.add_argument('--figure', type=Path, help='Save the diagnostic figure here.')
    parser.add_argument('--show', action='store_true', help='Display the diagnostic figure.')
    parser.add_argument('--log-level', type=int, default=logging.WARNING)
    return parser


def _list_types(registry: PluginRegistry) -> None:
    for title, chooser in (
        ('Diffraction file readers', registry.diffraction_file_readers),
        ('Probe position file readers', registry.probe_position_file_readers),
    ):
        print(f'{title}:')

        for plugin in chooser:
            print(f'  {plugin.simple_name:32s} {plugin.display_name}')


def _reader(chooser: PluginChooser[_ReaderT], name: str, kind: str) -> _ReaderT:
    plugin = chooser.find_plugin(name)

    if plugin is None:
        raise SystemExit(f'No {kind} reader named "{name}"; see --list-types.')

    return plugin.strategy


def _accumulate(dataset: DiffractionDataset, bin_size_px: int) -> DiffractionMoments:
    """The dataset's moments, one array at a time, as the frames were recorded."""
    extent = dataset.get_metadata().detector_extent
    accumulator = DiffractionMomentAccumulator(
        extent.height_px,
        extent.width_px,
        bad_pixels=dataset.get_bad_pixels(),
        binning=BinningStep(bin_size_x=bin_size_px, bin_size_y=bin_size_px),
    )

    for array in dataset:
        accumulator.add(array.get_indexes(), array.get_patterns())

    moments = accumulator.get_moments()

    if moments is None:
        raise SystemExit('The diffraction file holds no patterns.')

    return moments


def _correct_beam_decay(
    moments: DiffractionMoments, positions: ProbePositionSequence
) -> DiffractionMoments:
    """Correct the totals for beam decay by the I0 the positions carry, if any."""
    i0_counts_by_index = positions.get_probe_photon_counts_by_index()

    if not i0_counts_by_index:
        return moments

    return moments.normalize_by_i0(
        numpy.fromiter(i0_counts_by_index.keys(), dtype=numpy.intp),
        numpy.fromiter(i0_counts_by_index.values(), dtype=numpy.float64),
    )


def _image_geometry(positions: ProbePositionSequence, pixel_m: float | None) -> ObjectGeometry:
    """A canvas over the scan for the bright-field images, a few pixels of margin each side."""
    scan = calculate_scan_geometry(positions)

    if scan is None:
        raise SystemExit('The position file holds no positions.')

    if pixel_m is None:
        xy_m = numpy.column_stack(
            (positions.get_coordinates_x_m(), positions.get_coordinates_y_m())
        )
        distances_m, _ = cKDTree(xy_m).query(xy_m, k=2)
        pixel_m = float(numpy.median(distances_m[:, 1])) / 2.0

    if not pixel_m > 0.0:
        raise SystemExit('Cannot size the bright-field images; pass --image-pixel-m.')

    margin_px = 4
    return ObjectGeometry(
        width_px=math.ceil(scan.width_m / pixel_m) + 2 * margin_px,
        height_px=math.ceil(scan.height_m / pixel_m) + 2 * margin_px,
        pixel_width_m=pixel_m,
        pixel_height_m=pixel_m,
        center_x_m=scan.center_x_m,
        center_y_m=scan.center_y_m,
    )


def _describe(candidate: OrientationCandidate) -> str:
    """A reading as its defocus and the orientation steps it calls for."""
    orientation = candidate.orientation
    steps = [
        name
        for name, applied in (
            ('hflip', orientation.hflip),
            ('vflip', orientation.vflip),
            ('transpose', orientation.transpose),
        )
        if applied
    ]
    return (
        f'defocus {candidate.defocus_distance_m * 1e3:+.4g} mm:'
        f' apply {", ".join(steps) if steps else "no orientation steps"}'
    )


def _report(transform: ParallaxTransform) -> None:
    matrix = transform.matrix_m2
    spread_m = float(numpy.std(transform.shifts_m))
    print(f'Bright-field bins fitted: {transform.num_bins}')
    print('Fitted map s = a + M q, (x, y) [m^2]:')
    print(f'  M = [[{matrix[0, 0]:+.4e}, {matrix[0, 1]:+.4e}],')
    print(f'       [{matrix[1, 0]:+.4e}, {matrix[1, 1]:+.4e}]]')
    print(
        f'  residual rms {transform.residual_rms_m * 1e9:.3g} nm against a shift spread of'
        f' {spread_m * 1e9:.3g} nm'
    )
    print(
        f'Defocus |z| {transform.defocus_magnitude_m * 1e3:.4g} mm; depth of focus'
        f' {transform.depth_of_focus_m * 1e3:.4g} mm; Fresnel number'
        f' {transform.fresnel_number:.3g}'
    )

    if transform.fresnel_number < 1.0:
        print(
            '  WARNING: within the depth of focus the bright-field images are not shifted'
            ' copies of one another, so this result is unreliable.'
        )

    print(f'Mirrored relative to the scan: {"yes" if transform.is_reflection else "no"}')

    try:
        correction = transform.position_correction()
    except ValueError as exc:
        print(f'  Could not decompose what is left once the orientation is removed: {exc}')
    else:
        components = correction.decompose()
        rotation_deg = math.degrees(components.rotation_rad)
        print(
            "Position correction, in pty-chi's convention (positive rotation turns x towards y):"
            f' rotation {rotation_deg:+.2f} deg, asymmetry {components.asymmetry:+.3f},'
            f' shear {math.degrees(components.shear_rad):+.2f} deg'
        )
        # Laid out as the probe-position editor's Transformation box lays out its entries.
        print(f"  x' = {correction.a00:+.6f} x {correction.a01:+.6f} y + 0")
        print(f"  y' = {correction.a10:+.6f} x {correction.a11:+.6f} y + 0")

        if abs(rotation_deg) > 2.0 or abs(components.asymmetry) > 0.05:
            print(
                '  A rotation beyond 2 deg or an asymmetry beyond 0.05 calls for entering this as'
                ' the probe-position affine transform.'
            )

    print('Readings, equally good -- the sign of the defocus picks one:')

    for candidate in transform.orientation_candidates():
        print(f'  {_describe(candidate)}')


def _plot(transform: ParallaxTransform, *, figure_path: Path | None, show: bool) -> None:
    import matplotlib

    if not show:
        matplotlib.use('Agg')

    import matplotlib.pyplot as plt

    q = transform.q_per_m
    measured = transform.shifts_m
    fitted = transform.offset_m + q @ transform.matrix_m2.T
    residual = measured - fitted
    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 5.5), layout='constrained')

    def draw(axes: Any, vectors: Any, scale: float, **style: Any) -> None:
        axes.quiver(
            q[:, 0],
            q[:, 1],
            vectors[:, 0],
            vectors[:, 1],
            angles='xy',
            scale_units='xy',
            scale=scale,
            **style,
        )

    # One scale per panel, in q units: the longest arrow spans about one bin spacing.
    extent = numpy.ptp(q, axis=0)
    spacing = float(numpy.min(extent[extent > 0.0])) / math.sqrt(transform.num_bins)
    longest_shift = float(numpy.max(numpy.hypot(*numpy.vstack((measured, fitted)).T)))
    longest_residual = float(numpy.max(numpy.hypot(*residual.T)))

    draw(left, fitted, longest_shift / spacing, color='C1', label='fitted')
    draw(
        left,
        measured,
        longest_shift / spacing,
        facecolor='none',
        edgecolor='C0',
        linewidth=1.0,
        label='measured',
    )
    left.set_title(f'Image shift per bright-field bin\n(longest {longest_shift * 1e9:.3g} nm)')
    draw(right, residual, max(longest_residual, 1e-30) / spacing, color='C3')
    right.set_title(
        f'Residual, rms {transform.residual_rms_m * 1e9:.3g} nm'
        f'\n(longest {longest_residual * 1e9:.3g} nm, magnified)'
    )

    for axes in (left, right):
        axes.set_xlabel('q_x [1/m]')
        axes.set_ylabel('q_y [1/m]')
        axes.set_aspect('equal')
        axes.margins(0.15)

    left.legend()
    first, second = transform.orientation_candidates()
    figure.suptitle(f'{_describe(first)}   or   {_describe(second)}')

    if figure_path is not None:
        figure.savefig(figure_path, dpi=150)

    if show:
        plt.show()


def main() -> int:
    args = _build_parser().parse_args()
    logging.basicConfig(level=args.log_level)
    registry = PluginRegistry.load_plugins()

    if args.list_types:
        _list_types(registry)
        return 0

    for flag in ('patterns_type', 'patterns', 'positions_type', 'positions'):
        if getattr(args, flag) is None:
            raise SystemExit(f'--{flag.replace("_", "-")} is required; see --help.')

    dataset = _reader(registry.diffraction_file_readers, args.patterns_type, 'diffraction').read(
        args.patterns
    )
    positions = _reader(registry.probe_position_file_readers, args.positions_type, 'position').read(
        args.positions
    )
    metadata = dataset.get_metadata()

    energy_eV = args.energy_eV or metadata.photon_energy_eV  # noqa: N806
    distance_m = args.detector_distance_m or metadata.detector_distance_m
    pixel_geometry = metadata.detector_pixel_geometry

    if args.pixel_size_m is not None:
        pixel_geometry = PixelGeometry(
            width_m=args.pixel_size_m, height_m=args.pixel_height_m or args.pixel_size_m
        )

    missing = [
        flag
        for flag, value in (
            ('--energy-eV', energy_eV),
            ('--detector-distance-m', distance_m),
            ('--pixel-size-m', pixel_geometry),
        )
        if not value
    ]

    if energy_eV is None or distance_m is None or pixel_geometry is None or missing:
        raise SystemExit(f'The file does not state them; pass {", ".join(missing)}.')

    wavelength_m = energy_eV_to_wavelength_m(energy_eV)
    moments = _correct_beam_decay(_accumulate(dataset, args.bin_size_px), positions)
    height_px, width_px = moments.mean_frame.shape
    probe_geometry = ProbeGeometry.from_far_field(
        pixel_geometry,
        ImageExtent(width_px=width_px, height_px=height_px),
        wavelength_m=wavelength_m,
        distance_m=distance_m,
    )
    transform = estimate_parallax_transform(
        moments,
        positions,
        _image_geometry(positions, args.image_pixel_m),
        probe_geometry,
        photon_wavelength_m=wavelength_m,
        bright_field_fraction=args.bright_field_fraction,
    )

    if transform is None:
        print(
            'Too few bright-field bins or positions for a fit; try a smaller --bin-size-px'
            ' or a lower --bright-field-fraction.'
        )
        return 1

    _report(transform)

    if args.figure is not None or args.show:
        _plot(transform, figure_path=args.figure, show=args.show)

    return 0


if __name__ == '__main__':
    sys.exit(main())

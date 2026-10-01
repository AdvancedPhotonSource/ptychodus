#!/usr/bin/env python
"""Reconstruct a range of APS 31-ID-E LamNI projections from one experiment directory.

A laminography tomogram is one scan per projection -- 1440 and up, split into interleaved
sub-tomograms -- and ``tomography_scannumbers.txt`` is the index that ties each scan number
to its rotation angle and its place in the tomogram. This driver walks that table and runs
``reconstruct_lamni``'s per-scan pipeline over the selected projections, so a tomogram is one
command rather than a thousand.

Each scan lands in its own standard-layout directory under
``<experiment>/analysis/ptychodus/<label>/S<NNNNN>/``: ``diffraction.h5``,
``ptychi_options.json`` recording the options that actually ran, per-epoch
``product.NNNNNN.h5`` checkpoints, and the final ``product.h5``. Every product carries its
own ``tomography_angle_deg``, taken from the encoder column of the scan table, which is what
lets the resulting stack be fed to a tomographic solver.

Affine scanner calibration
--------------------------

With ``--affine-transform-after N``, the first N projections reconstruct with pty-chi's
position correction enabled, then their measured and refined positions are fitted jointly by
:func:`~ptychodus.api.affine.estimate_affine_transform` for one shared linear part. Every
later projection has position correction switched off and that transform applied to its
measured positions instead. The per-pair translations the fit reports are deliberately not
applied: a separate scan has a separate stage origin, so only the linear part carries over.

Scope and limitations
---------------------

- One tomogram per run, selected by label. Projections are reconstructed in scan-number
  order, which is acquisition order, because both probe chaining and the affine phase depend
  on it.
- The crop region is resolved once, from the first selected scan, and reused for every scan.
  A per-scan beam center would put each projection on a different grid and break the
  warm-start probe.
- Repeated acquisitions of one ``(sub-tomogram, projection)`` are all reconstructed. The scan
  table records a re-run as a further row, and which one supersedes the other is not
  something the table says.
- pty-chi runs in this process, once per scan, and GPU memory is not reclaimed between
  scans. Resume a run that ends early with ``--skip-existing``.
- No GPU selection. Choose a device with ``CUDA_VISIBLE_DEVICES`` in the environment.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy


from ptychodus.api.affine import (
    AffineFitResult,
    AffineTransform,
    estimate_affine_transform,
    transform_probe_positions,
)
from ptychodus.api.assemble import assemble_dataset, summarize_dataset
from ptychodus.api.constants import energy_eV_to_wavelength_m
from ptychodus.api.diffraction import BadPixels, BeamCenter, CropRegion, DiffractionDataset
from ptychodus.api.exit_codes import ExitCode
from ptychodus.api.geometry import ImageExtent, PixelGeometry
from ptychodus.api.io import StandardFileLayout
from ptychodus.api.object import compute_object_geometry
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.preprocess.diffraction import (
    DiffractionPrepPipeline,
    FilterValuesStep,
    estimate_beam_center,
)
from ptychodus.api.probe import Probe, ProbeGeometry, ProbeSequence
from ptychodus.api.probe_positions import ProbePositionSequence
from ptychodus.api.product import Product, ProductMetadata
from ptychodus.api.reconstruct import prepare_reconstruct_input
from ptychodus.api.simulate.object import generate_random_object
from ptychodus.api.simulate.probe import (
    generate_coherent_probe_modes,
    generate_fresnel_zone_plate_probe,
    generate_incoherent_probe_modes,
)
from ptychodus.cli import DirectoryType
from ptychodus.cli._reconstruct_common import (
    CancellationToken,
    add_ptychi_options_argument,
    install_signal_handlers,
    is_main_process,
    load_ptychi_options,
    run_reconstruction,
    save_assembled_diffraction,
)
from ptychodus.plugins.aps31id_lamni._scan_table import (
    APS31IDEScanRecord,
    read_aps31ide_scan_table,
)

logger = logging.getLogger('reconstruct_batch_lamni')

DIFFRACTION_READER = 'APS_LamNI'
POSITION_READER = 'APS_LamNI'

DEFAULT_FZP_PRESET = 'APS_LamNI'

# Scans live under eiger_4 in blocks of a thousand.
_SCAN_BLOCK_SIZE = 1000


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f'"{text}" must be at least 1!')
    return value


def _resolve(
    quantity: str,
    override: float | None,
    from_file: float | None,
    fallback: float | None,
    flag: str,
) -> float:
    """Pick a geometry value from the command line, the file, or a fallback, and say which.

    Reporting the source is the point: a run that silently substituted a built-in
    default for a value the instrument actually recorded would be indistinguishable
    from one that read it.
    """
    if override is not None:
        logger.debug('%s: %g (from %s)', quantity, override, flag)
        return override

    if from_file is not None:
        logger.debug('%s: %g (from the diffraction file)', quantity, from_file)
        return from_file

    if fallback is not None:
        logger.warning(
            '%s: %g (built-in default for this instrument; the file recorded none). '
            'Pass %s if this scan differs.',
            quantity,
            fallback,
            flag,
        )
        return fallback

    raise ValueError(f'{quantity} is not in the file and has no default; pass {flag}.')


def _invalid_count_threshold(dtype: numpy.dtype) -> int | None:
    """Count at and above which a pixel is invalid rather than merely bright.

    Dectris detectors flag dead, masked and saturated pixels with the maximum value the
    pattern dtype can hold -- 4294967295 on a uint32 Eiger. Left in place those markers
    dominate every statistic computed from the frame: the beam-center estimator rejects
    the real beam as noise and silently returns the detector midpoint instead.
    """
    if numpy.issubdtype(dtype, numpy.integer):
        return int(numpy.iinfo(dtype).max)

    return None


@dataclass(frozen=True)
class _Layout:
    """The experiment directories this driver reads from and writes to."""

    experiment_root: Path
    data_dir: Path

    @classmethod
    def from_argument(cls, directory: Path) -> _Layout:
        """Accept either the experiment root or the ``data`` directory inside it."""
        if (directory / 'data' / 'dat-files').is_dir():
            return cls(experiment_root=directory, data_dir=directory / 'data')

        if (directory / 'dat-files').is_dir():
            return cls(experiment_root=directory.parent, data_dir=directory)

        raise ValueError(
            f'{directory} is not a 31-ID-E experiment directory: expected a "dat-files" '
            'subdirectory either directly inside it or under a "data" subdirectory.'
        )

    @property
    def scan_table_file(self) -> Path:
        return self.data_dir / 'dat-files' / 'tomography_scannumbers.txt'

    def position_file(self, scan_no: int) -> Path:
        return self.data_dir / 'scan_positions' / f'scan_{scan_no:05d}.dat'

    def diffraction_file(self, scan_no: int) -> Path:
        """Locate the one data file of a scan's detector series.

        The series is written beside a ``*_master_*`` index file that holds no patterns, and
        the stem differs between acquisition software versions, so the directory is globbed
        rather than assumed. An ambiguous match is an error: picking one arbitrarily would
        reconstruct an unpredictable subset.
        """
        block_start = (scan_no // _SCAN_BLOCK_SIZE) * _SCAN_BLOCK_SIZE
        block = f'S{block_start:05d}-{block_start + _SCAN_BLOCK_SIZE - 1:05d}'
        scan_dir = self.data_dir / 'eiger_4' / block / f'S{scan_no:05d}'

        candidates = sorted(
            path for path in scan_dir.glob(f'*_{scan_no:05d}_*.h5') if '_master_' not in path.name
        )

        if not candidates:
            raise FileNotFoundError(f'No diffraction file for scan {scan_no} in {scan_dir}.')

        if len(candidates) > 1:
            names = ', '.join(path.name for path in candidates)
            raise FileNotFoundError(
                f'Scan {scan_no} has {len(candidates)} candidate diffraction files in '
                f'{scan_dir}: {names}.'
            )

        return candidates[0]

    def output_directory(self, output_root: Path, record: APS31IDEScanRecord) -> Path:
        return output_root / record.label / f'S{record.scan_no:05d}'


def _select_label(records: list[APS31IDEScanRecord], requested: str | None) -> str:
    """Choose which tomogram to reconstruct, reporting what else the table holds."""
    counts = Counter(record.label for record in records)

    if not counts:
        raise ValueError('The scan table holds no parsable rows.')

    summary = ', '.join(f'{label} ({count} scans)' for label, count in counts.most_common())

    if requested is not None:
        if requested not in counts:
            raise ValueError(f'No scans labeled "{requested}". The table holds: {summary}.')

        logger.info(
            'Label: %s (%d scans). The table also holds: %s', requested, counts[requested], summary
        )
        return requested

    label, count = counts.most_common(1)[0]
    logger.warning(
        'No --label given; taking "%s", the largest tomogram in the table (%d scans). '
        'The table holds: %s',
        label,
        count,
        summary,
    )
    return label


def _select_records(
    records: list[APS31IDEScanRecord], args: argparse.Namespace, label: str
) -> list[APS31IDEScanRecord]:
    """Filter the table to the requested projections, in acquisition order."""
    selected = [record for record in records if record.label == label]

    if args.sub_tomogram is not None:
        selected = [r for r in selected if r.subtomo_no == args.sub_tomogram]

    if args.first_projection is not None:
        selected = [r for r in selected if r.measurement_id >= args.first_projection]

    if args.last_projection is not None:
        selected = [r for r in selected if r.measurement_id <= args.last_projection]

    if args.projection_stride > 1:
        projections = sorted({r.measurement_id for r in selected})
        kept = set(projections[:: args.projection_stride])
        selected = [r for r in selected if r.measurement_id in kept]

    # Scan number is acquisition order, which is what probe chaining and the affine phase
    # both assume; the table itself is already in that order but is not guaranteed to be.
    selected.sort(key=lambda record: record.scan_no)
    return selected


def _write_affine_fit(file_path: Path, result: AffineFitResult) -> None:
    """Record the fitted transform beside the tomogram it was fitted from."""
    components = result.components
    transform = result.linear_transform
    payload = {
        'scale': components.scale,
        'asymmetry': components.asymmetry,
        'rotation_deg': math.degrees(components.rotation_rad),
        'shear_deg': math.degrees(components.shear_rad),
        'rms_residual_m': result.rms_residual_m,
        'linear_transform': [
            [transform.a00, transform.a01, transform.a02],
            [transform.a10, transform.a11, transform.a12],
        ],
        'pairs': [
            {
                'translation_x_m': pair.translation_x_m,
                'translation_y_m': pair.translation_y_m,
                'num_points': pair.num_points,
                'num_inliers': pair.num_inliers,
                'rms_residual_m': pair.rms_residual_m,
            }
            for pair in result.pairs
        ],
    }
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(payload, indent=2))


def _resolve_crop_region(
    args: argparse.Namespace,
    raw_dataset: DiffractionDataset,
    value_filter: FilterValuesStep | None,
    bad_pixels: BadPixels | None,
) -> CropRegion | None:
    """Resolve one crop region for the whole run, from the first selected scan.

    Every projection must land on the same grid: the warm-start probe is carried from one
    scan to the next, and a stack of projections reconstructed on differing grids cannot be
    combined. So the beam center is settled once here rather than per scan.
    """
    if args.crop_extent_px is None:
        return None

    metadata = raw_dataset.get_metadata()

    if args.beam_center_x_px is not None and args.beam_center_y_px is not None:
        beam_center = BeamCenter(x_px=args.beam_center_x_px, y_px=args.beam_center_y_px)
        center_source = '--beam-center-{x,y}-px'
    elif metadata.beam_center is not None:
        beam_center = metadata.beam_center
        center_source = 'the diffraction file'
    else:
        # Reads every pattern of this scan, so it is reached only when no center was
        # supplied by flag or file. summarize_dataset inpaints the bad pixels that
        # estimate_beam_center requires the caller to have handled.
        logger.info('Summarizing the first scan to estimate the beam center')
        summary = summarize_dataset(raw_dataset, bad_pixels=bad_pixels)
        mean_pattern = (
            summary.mean_pattern
            if value_filter is None
            else value_filter.apply(summary.mean_pattern)
        )
        beam_center = estimate_beam_center(mean_pattern)
        center_source = 'an estimate over the whole scan'

    logger.info('Beam center: (%d, %d) from %s', beam_center.x_px, beam_center.y_px, center_source)

    extent = ImageExtent(width_px=args.crop_extent_px, height_px=args.crop_extent_px)

    if extent == metadata.detector_extent:
        return None

    region = CropRegion.from_center_extent(beam_center, extent)

    # from_center_extent does not clip. Silently clamping would quietly reconstruct a
    # different region than asked for, so an overhanging crop is an error.
    if region.clamp_to_detector_extent(metadata.detector_extent) != region:
        raise ValueError(
            f'A {args.crop_extent_px}px crop about ({beam_center.x_px}, '
            f'{beam_center.y_px}) runs off the '
            f'{metadata.detector_extent.width_px}x{metadata.detector_extent.height_px} '
            f'detector (x={region.x_range} y={region.y_range}). '
            'Give a smaller --crop-extent-px or an explicit beam center.'
        )

    logger.info('Cropping to x=%s y=%s', region.x_range, region.y_range)
    return region


@dataclass(frozen=True)
class _ScanResult:
    """What one reconstructed projection contributes to the rest of the run."""

    measured_positions: ProbePositionSequence
    reconstructed_positions: ProbePositionSequence
    probes: ProbeSequence


def _reconstruct_one_scan(
    args: argparse.Namespace,
    layout: _Layout,
    registry: PluginRegistry,
    record: APS31IDEScanRecord,
    output_directory: Path,
    *,
    read_region: CropRegion | None,
    bad_pixels: BadPixels | None,
    warm_start_probe: Probe | None,
    transform: AffineTransform | None,
    cancellation: CancellationToken,
) -> _ScanResult | None:
    """Reconstruct one projection into its own standard-layout directory.

    Returns `None` for a scan cancelled before it produced anything, which contributes
    neither a probe to chain forward nor a position pair to the affine fit.
    """
    diffraction_file_path = layout.diffraction_file(record.scan_no)
    position_file_path = layout.position_file(record.scan_no)

    diffraction_reader = registry.diffraction_file_readers.get_strategy_by_name(DIFFRACTION_READER)
    position_reader = registry.probe_position_file_readers.get_strategy_by_name(POSITION_READER)

    logger.info('Reading diffraction from %s', diffraction_file_path)
    raw_dataset = diffraction_reader.read(diffraction_file_path)
    metadata = raw_dataset.get_metadata()

    detector_distance_m = _resolve(
        'Detector distance (m)',
        args.detector_distance_m,
        metadata.detector_distance_m,
        None,
        '--detector-distance-m',
    )
    probe_energy_eV = _resolve(  # noqa: N806
        'Probe energy (eV)',
        args.probe_energy_eV,
        metadata.probe_energy_eV,
        None,
        '--probe-energy-eV',
    )
    if metadata.detector_pixel_geometry is None:
        raise ValueError('The diffraction file records no detector pixel size!')

    detector_pixel_size_m = metadata.detector_pixel_geometry.width_m
    logger.debug('Detector pixel size (m): %g (from the diffraction file)', detector_pixel_size_m)
    raw_pixel_geometry = PixelGeometry(
        width_m=detector_pixel_size_m, height_m=detector_pixel_size_m
    )

    max_valid_count = (
        _invalid_count_threshold(metadata.pattern_dtype)
        if args.max_valid_count is None
        else args.max_valid_count
    )
    pipeline = (
        None
        if max_valid_count is None
        else DiffractionPrepPipeline(
            steps=(FilterValuesStep(lower_bound=0, upper_bound=max_valid_count),)
        )
    )

    assembled_data = assemble_dataset(
        raw_dataset,
        pipeline,
        bad_pixels=bad_pixels,
        raw_pixel_geometry=raw_pixel_geometry,
        read_region=read_region,
        total_counts_lower_bound=args.min_total_counts,
    )
    num_patterns = assembled_data.get_num_patterns()
    num_raw_patterns = sum(metadata.num_patterns_per_array)

    if num_patterns != num_raw_patterns:
        logger.warning(
            'Assembled %d of %d patterns; %d were filtered or failed to load.',
            num_patterns,
            num_raw_patterns,
            num_raw_patterns - num_patterns,
        )

    save_assembled_diffraction(
        logger, output_directory, assembled_data, skip=args.no_save_diffraction
    )

    measured_positions = position_reader.read(position_file_path)
    logger.info('Read %d probe positions from %s', len(measured_positions), position_file_path)

    if transform is None:
        positions = measured_positions
    else:
        positions = ProbePositionSequence(
            [*transform_probe_positions(measured_positions, transform)]
        )

    probe_wavelength_m = energy_eV_to_wavelength_m(probe_energy_eV)
    probe_geometry = ProbeGeometry.from_far_field(
        assembled_data.get_pixel_geometry(),
        assembled_data.get_image_extent(),
        wavelength_m=probe_wavelength_m,
        distance_m=detector_distance_m,
    )

    rng = numpy.random.default_rng(args.seed)

    if warm_start_probe is None:
        logger.info('Simulating the initial probe from zone plate preset "%s"', args.fzp_preset)
        zone_plate = registry.fresnel_zone_plates.get_strategy_by_name(args.fzp_preset)
        probe = generate_fresnel_zone_plate_probe(
            probe_geometry,
            zone_plate,
            probe_wavelength_m=probe_wavelength_m,
            defocus_distance_m=args.fzp_defocus_m,
        )
    else:
        array = warm_start_probe.get_array()

        # The crop is fixed for the whole run, so a mismatch here means the geometry moved
        # between projections rather than that the user changed the crop.
        if array.shape[-2:] != (probe_geometry.height_px, probe_geometry.width_px):
            raise ValueError(
                f'Warm-start probe extent {array.shape[-2]}x{array.shape[-1]} does not match '
                f'the patterns {probe_geometry.height_px}x{probe_geometry.width_px}.'
            )

        probe = Probe(array=array, pixel_geometry=probe_geometry.get_pixel_geometry())

    if args.num_probe_modes > 1 and probe.get_array().shape[0] < args.num_probe_modes:
        # Geometric weights: each successive incoherent mode carries half the power of
        # the one before it. A warm-started probe already carries its modes.
        weights = [0.5**imode for imode in range(args.num_probe_modes)]
        probe = generate_incoherent_probe_modes(rng, probe, weights)

    probe_sequence = generate_coherent_probe_modes(
        rng,
        probe,
        num_cmodes=args.num_opr_modes,
        num_diffraction_patterns=num_patterns,
    )

    object_geometry = compute_object_geometry(
        positions, probe_geometry, padding_px=args.object_padding_px
    )

    # Deviations = 0 gives a flat unit-amplitude, zero-phase field.
    object_ = generate_random_object(
        rng,
        object_geometry,
        amplitude_mean=1.0,
        amplitude_deviation=0.0,
        phase_mean=0.0,
        phase_deviation_tr=0.0,
        blur_deviation_px=0.0,
    )

    probe_photon_count = float(assembled_data.get_probe_photon_count())

    product = Product(
        metadata=ProductMetadata(
            name=f'scan{record.scan_no:05d}_{record.label}',
            comments=str(record),
            detector_distance_m=detector_distance_m,
            probe_energy_eV=probe_energy_eV,
            probe_photon_count=probe_photon_count,
            exposure_time_s=float(metadata.exposure_time_s or 0.0),
            mass_attenuation_m2_kg=0.0,
            tomography_angle_deg=record.encoder_angle_deg,
        ),
        probe_positions=positions,
        probes=probe_sequence,
        object_=object_,
        losses=[],
    )

    options = load_ptychi_options(args.ptychi_options_file)

    if transform is not None:
        # The transform now supplies the correction that per-scan refinement was measuring.
        options.probe_position_options.optimizable = False

    reconstruct_input = prepare_reconstruct_input(assembled_data, product)
    final_product = run_reconstruction(
        logger,
        reconstruct_input,
        options,
        output_directory,
        num_sync_epochs=args.num_sync_epochs,
        cancellation=cancellation,
    )

    if final_product is None:
        return None

    return _ScanResult(
        # The measured positions as the reconstruction actually saw them: pattern-paired,
        # so they index-match the refined positions coming back out.
        measured_positions=reconstruct_input.product.probe_positions,
        reconstructed_positions=final_product.probe_positions,
        probes=final_product.probes,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Reconstruct a range of APS 31-ID-E LamNI projections from one experiment '
            'directory via the ptychodus api.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'experiment_directory',
        type=DirectoryType(must_exist=True),
        help='Experiment root directory, or the "data" directory inside it.',
    )
    parser.add_argument(
        '--output-directory',
        type=DirectoryType(must_exist=False),
        default=None,
        help=(
            'Parent of the per-scan <label>/S<NNNNN>/ output directories. '
            'Defaults to <experiment>/analysis/ptychodus.'
        ),
    )
    parser.add_argument(
        '--label',
        default=None,
        help='Tomogram to reconstruct. Defaults to the largest one in the scan table.',
    )
    parser.add_argument(
        '--first-projection',
        type=int,
        default=None,
        help='Lowest projection index to reconstruct. Omit for all.',
    )
    parser.add_argument(
        '--last-projection',
        type=int,
        default=None,
        help='Highest projection index to reconstruct. Omit for all.',
    )
    parser.add_argument(
        '--sub-tomogram',
        type=int,
        default=None,
        help='Reconstruct only this sub-tomogram. Omit for all of them.',
    )
    parser.add_argument(
        '--projection-stride',
        type=_positive_int,
        default=1,
        help='Reconstruct every Nth distinct projection index.',
    )
    add_ptychi_options_argument(parser)
    parser.add_argument(
        '--affine-transform-after',
        type=_positive_int,
        default=None,
        help=(
            'Fit one shared affine transform from the first N reconstructions, then disable '
            'pty-chi position correction and apply it to the measured positions instead.'
        ),
    )
    parser.add_argument(
        '--no-warm-start-probe',
        action='store_true',
        help='Rebuild the probe from the zone-plate model for every scan.',
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip a scan whose product.h5 already exists, so an interrupted run resumes.',
    )
    parser.add_argument(
        '--continue-on-error',
        action='store_true',
        help='Log and continue when a scan fails instead of aborting the run.',
    )
    parser.add_argument(
        '--no-save-diffraction',
        action='store_true',
        help='Skip diffraction.h5. The assembled patterns can run to several GB per scan.',
    )
    parser.add_argument(
        '--crop-extent-px',
        type=_positive_int,
        default=None,
        help='Square crop side in raw detector pixels, about the beam center. Omit to skip.',
    )
    parser.add_argument(
        '--beam-center-x-px',
        type=int,
        default=None,
        help='Beam center column. File value, then an estimate from the data, otherwise.',
    )
    parser.add_argument(
        '--beam-center-y-px',
        type=int,
        default=None,
        help='Beam center row. File value, then an estimate from the data, otherwise.',
    )
    parser.add_argument(
        '--detector-distance-m',
        type=float,
        default=None,
        help='Sample-to-detector distance. Overrides the file.',
    )
    parser.add_argument(
        '--probe-energy-eV',
        type=float,
        default=None,
        help='Probe energy in electron volts. Overrides the file.',
    )
    parser.add_argument(
        '--min-total-counts',
        type=int,
        default=None,
        help='Drop patterns whose total counts fall below this, with their positions.',
    )
    parser.add_argument(
        '--max-valid-count',
        type=int,
        default=None,
        help=(
            'Zero pixels at or above this count. Defaults to the pattern dtype maximum, '
            'which is how Dectris detectors mark invalid pixels.'
        ),
    )
    parser.add_argument(
        '--bad-pixels-file',
        type=Path,
        default=None,
        help='Bad-pixel mask covering the full, uncropped detector.',
    )
    parser.add_argument(
        '--bad-pixels-file-type',
        default='NPY_Bad_Pixels',
        help='Name of the bad-pixel file reader plugin.',
    )
    parser.add_argument(
        '--probe-file',
        type=Path,
        default=None,
        help='Initial probe for the first scan, normally a product HDF5 from an earlier run.',
    )
    parser.add_argument(
        '--probe-file-type',
        default='HDF5',
        help='Name of the probe file reader plugin; HDF5 reads a ptychodus product.',
    )
    parser.add_argument(
        '--fzp-preset',
        default=DEFAULT_FZP_PRESET,
        help='FresnelZonePlate plugin preset for the model probe.',
    )
    parser.add_argument(
        '--fzp-defocus-m',
        type=float,
        default=800e-6,
        help='Defocus from the zone-plate focal plane.',
    )
    parser.add_argument(
        '--num-probe-modes',
        type=_positive_int,
        default=1,
        help='Incoherent probe modes.',
    )
    parser.add_argument(
        '--num-opr-modes',
        type=_positive_int,
        default=1,
        help='Coherent (OPR) probe modes; 1 disables OPR.',
    )
    parser.add_argument(
        '--object-padding-px',
        type=int,
        default=64,
        help='Object-canvas margin added to each side of the scan bounding box.',
    )
    parser.add_argument('--seed', type=int, default=0, help='RNG seed for the initial guesses.')
    parser.add_argument(
        '--num-sync-epochs',
        type=_positive_int,
        default=100,
        help='Epochs between reconstructor sync points and progress logs.',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Report the resolved scan plan and stop without reading or reconstructing data.',
    )
    parser.add_argument(
        '--log-level',
        default=logging.INFO,
        type=int,
        help='Python logging level.',
    )
    return parser


def main() -> ExitCode:
    args = _build_parser().parse_args()

    logging.basicConfig(
        level=args.log_level,
        stream=sys.stderr,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )

    # Installed before the first long operation, so a cancel during the file reads is
    # already honored. Nothing earlier than this is interruptible.
    cancellation = install_signal_handlers()

    layout = _Layout.from_argument(args.experiment_directory)
    output_root = (
        layout.experiment_root / 'analysis' / 'ptychodus'
        if args.output_directory is None
        else args.output_directory
    )

    logger.info('Reading the scan table from %s', layout.scan_table_file)
    records = read_aps31ide_scan_table(layout.scan_table_file)
    label = _select_label(records, args.label)
    selected = _select_records(records, args, label)

    if not selected:
        logger.error('No scans matched the requested projection range.')
        return ExitCode.FAILURE

    logger.info(
        'Selected %d scan(s): %d..%d',
        len(selected),
        selected[0].scan_no,
        selected[-1].scan_no,
    )

    if args.dry_run:
        for record in selected:
            output_directory = layout.output_directory(output_root, record)

            try:
                diffraction_file_path: Path | str = layout.diffraction_file(record.scan_no)
            except FileNotFoundError as exc:
                diffraction_file_path = f'MISSING ({exc})'

            position_file_path = layout.position_file(record.scan_no)
            position_note = '' if position_file_path.is_file() else ' MISSING'
            logger.info(
                'scan %05d subtomo %d projection %d angle %+.4f deg\n'
                '  diffraction: %s\n'
                '  positions:   %s%s\n'
                '  output:      %s',
                record.scan_no,
                record.subtomo_no,
                record.measurement_id,
                record.encoder_angle_deg,
                diffraction_file_path,
                position_file_path,
                position_note,
                output_directory,
            )

        logger.info('Dry run: %d scan(s) planned. Nothing read, nothing written.', len(selected))
        return ExitCode.SUCCESS

    registry = PluginRegistry.load_plugins()

    if args.affine_transform_after is not None:
        # A fit against positions that were never refined returns the identity, which would
        # look like a successful calibration while correcting nothing.
        probe_options = load_ptychi_options(args.ptychi_options_file).probe_position_options

        if not probe_options.optimizable:
            logger.error(
                '--affine-transform-after needs pty-chi position correction enabled for the '
                'calibration scans, but the supplied options disable it. The fit would see '
                'unrefined positions and return the identity transform.'
            )
            return ExitCode.FAILURE

    bad_pixels: BadPixels | None = None

    if args.bad_pixels_file is not None:
        logger.info('Reading bad pixels from %s', args.bad_pixels_file)
        bad_pixels_reader = registry.bad_pixels_file_readers.get_strategy_by_name(
            args.bad_pixels_file_type
        )
        bad_pixels = bad_pixels_reader.read(args.bad_pixels_file)

    # One crop for the whole run, settled from the first selected scan.
    first_dataset = registry.diffraction_file_readers.get_strategy_by_name(DIFFRACTION_READER).read(
        layout.diffraction_file(selected[0].scan_no)
    )
    first_metadata = first_dataset.get_metadata()
    max_valid_count = (
        _invalid_count_threshold(first_metadata.pattern_dtype)
        if args.max_valid_count is None
        else args.max_valid_count
    )

    # The same cut each scan's pipeline applies below, so the frame the beam center is
    # estimated from matches the frames that will be reconstructed.
    value_filter: FilterValuesStep | None = None

    if max_valid_count is not None:
        logger.info('Zeroing pixels at or above %d as invalid', max_valid_count)
        value_filter = FilterValuesStep(lower_bound=0, upper_bound=max_valid_count)

    read_region = _resolve_crop_region(args, first_dataset, value_filter, bad_pixels)
    del first_dataset

    warm_start_probe: Probe | None = None

    if args.probe_file is not None:
        logger.info('Reading the initial probe from %s', args.probe_file)
        probe_reader = registry.probe_file_readers.get_strategy_by_name(args.probe_file_type)
        warm_start_probe = probe_reader.read(args.probe_file).get_probe_no_opr()

    position_pairs: list[tuple[ProbePositionSequence, ProbePositionSequence]] = []
    transform: AffineTransform | None = None

    num_succeeded = 0
    skipped: list[int] = []
    failed: list[int] = []

    cancelled_after: int | None = None

    for scan_index, record in enumerate(selected, start=1):
        if cancellation.is_cancelled:
            cancelled_after = scan_index - 1
            logger.warning(
                'Cancelled: stopping before scan %05d, with %d of %d selected scan(s) done.',
                record.scan_no,
                scan_index - 1,
                len(selected),
            )
            break

        output_directory = layout.output_directory(output_root, record)

        if args.skip_existing and StandardFileLayout.PRODUCT.path(output_directory).is_file():
            logger.info(
                '[%d/%d] scan %05d: skipping, %s already exists',
                scan_index,
                len(selected),
                record.scan_no,
                StandardFileLayout.PRODUCT.path(output_directory),
            )
            skipped.append(record.scan_no)
            continue

        logger.info(
            '[%d/%d] scan %05d: subtomo %d, projection %d, angle %+.4f deg',
            scan_index,
            len(selected),
            record.scan_no,
            record.subtomo_no,
            record.measurement_id,
            record.encoder_angle_deg,
        )

        try:
            result = _reconstruct_one_scan(
                args,
                layout,
                registry,
                record,
                output_directory,
                read_region=read_region,
                bad_pixels=bad_pixels,
                warm_start_probe=warm_start_probe,
                transform=transform,
                cancellation=cancellation,
            )
        except Exception:
            if not args.continue_on_error:
                raise

            logger.exception('Scan %05d failed; continuing.', record.scan_no)
            failed.append(record.scan_no)
            continue

        if result is None:
            # Cancelled before this scan produced anything. It is neither a success nor a
            # failure, and the loop guard above ends the run on the next pass.
            cancelled_after = scan_index - 1
            break

        num_succeeded += 1

        if not args.no_warm_start_probe:
            warm_start_probe = result.probes.get_probe_no_opr()

        if args.affine_transform_after is not None and transform is None:
            position_pairs.append((result.measured_positions, result.reconstructed_positions))

            if len(position_pairs) >= args.affine_transform_after:
                logger.info(
                    'Fitting one affine transform across %d position pair(s)',
                    len(position_pairs),
                )
                fit = estimate_affine_transform(
                    position_pairs, rng=numpy.random.default_rng(args.seed)
                )
                logger.info('%s', fit)

                # Every rank fits the same transform from the same pairs and carries the
                # linear part forward below, so only the record of it is confined to one
                # process.
                if is_main_process():
                    affine_fit_file = output_root / label / 'affine_fit.json'
                    _write_affine_fit(affine_fit_file, fit)
                    logger.info('Wrote %s', affine_fit_file)

                # Only the linear part carries over. Each scan has its own stage origin, so
                # the per-pair translations the fit reports describe those scans alone.
                transform = fit.linear_transform
                logger.info(
                    'Position correction is off from here on; measured positions are '
                    'transformed by the fitted linear part instead.'
                )

    logger.info(
        'Done: %d reconstructed, %d skipped, %d failed, of %d selected.',
        num_succeeded,
        len(skipped),
        len(failed),
        len(selected),
    )

    if failed:
        logger.error('Failed scans: %s', ', '.join(f'{scan_no:05d}' for scan_no in failed))

    if cancelled_after is not None:
        # Reported ahead of the failures: the run was stopped, and whichever scans had
        # already failed is a detail of how far it got, not why it ended.
        logger.warning('Cancelled after %d of %d selected scan(s).', cancelled_after, len(selected))
        return ExitCode.CANCELLED

    return ExitCode.FAILURE if failed else ExitCode.SUCCESS


if __name__ == '__main__':
    sys.exit(main())

"""Reconstruct one LamNI projection into its own standard-layout directory.

Split from the driver because this is the half that pulls in a GPU stack, and it must not
be imported until the process that will run it has chosen its device. The driver plans a
run without importing any of this; a worker imports it only after setting
``CUDA_VISIBLE_DEVICES``.

Everything a scan produces is written here, into that scan's own directory. Nothing is
written above it. That is what lets scans run concurrently without coordinating: they
share no output path, so the usual single-writer rules have nothing to protect.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy
import yaml

from ptychodus.api.affine import AffineFitResult, estimate_affine_transform
from ptychodus.api.assemble import assemble_dataset
from ptychodus.api.constants import energy_eV_to_wavelength_m
from ptychodus.api.diffraction import BadPixels, CropRegion
from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.io import StandardFileLayout, save_product
from ptychodus.api.object import (
    LayerFillMode,
    Object,
    compute_object_geometry,
    compute_uniform_layer_spacing_m,
    resample_object_to_pixel_size,
    resize_object_layers,
)
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.preprocess.diffraction import DiffractionPrepPipeline, FilterValuesStep
from ptychodus.api.probe import ProbeGeometry, ProbeSequence, resample_probe_sequence
from ptychodus.api.probe_positions import ProbePositionSequence
from ptychodus.api.product import Product, ProductMetadata
from ptychodus.api.reconstruct import prepare_reconstruct_input
from ptychodus.api.simulate.object import generate_uniform_object
from ptychodus.api.simulate.probe import (
    generate_coherent_probe_modes,
    generate_fresnel_zone_plate_probe,
    generate_incoherent_probe_modes,
)
from ptychodus.cli._batch_lamni_settings import BatchSettings
from ptychodus.cli._reconstruct_common import (
    CancellationToken,
    invalid_count_threshold,
    resolve_quantity,
    run_reconstruction,
    save_assembled_diffraction,
)

__all__ = ['ScanRequest', 'reconstruct_one_scan', 'write_affine_fit']

logger = logging.getLogger('reconstruct_batch_lamni')

DIFFRACTION_READER = 'APS_LamNI'
POSITION_READER = 'APS_LamNI'


@dataclass(frozen=True)
class ScanRequest:
    """One scan's identity, as the scan table recorded it."""

    scan_no: int
    label: str
    encoder_angle_deg: float
    text: str


def write_affine_fit(file_path: Path, result: AffineFitResult) -> None:
    """Record the transform fitted from this scan's own measured and refined positions."""
    components = result.components
    transform = result.linear_transform
    payload = {
        'scale': components.scale,
        'asymmetry': components.asymmetry,
        'rotation_deg': float(numpy.degrees(components.rotation_rad)),
        'shear_deg': float(numpy.degrees(components.shear_rad)),
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
    # sort_keys=False: the payload is ordered to be read, coarsest description first.
    file_path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _estimate_and_write_affine_fit(
    output_directory: Path,
    measured: ProbePositionSequence,
    reconstructed: ProbePositionSequence,
    *,
    seed: int,
) -> None:
    """Fit this scan's own affine transform, if its positions were refined at all.

    Deliberately a single-scan fit, which is the noisy one: ``estimate_affine_transform``
    exists to pool several scans against one shared linear part, and a lone pair gives up
    that averaging. These files are the per-scan inputs to such a pooled fit, not a
    scanner calibration on their own.

    A failure here must not cost a finished reconstruction, so every way the fit can
    refuse -- too few matched points, collinear geometry, no consensus model -- is logged
    and dropped. ``product.h5`` is already on disk by this point.
    """
    displacement_m = numpy.hypot(
        numpy.asarray(reconstructed.get_coordinates_x_m())
        - numpy.asarray(measured.get_coordinates_x_m()),
        numpy.asarray(reconstructed.get_coordinates_y_m())
        - numpy.asarray(measured.get_coordinates_y_m()),
    )

    if not numpy.any(displacement_m > 0.0):
        # Position correction was off, so the refined positions are the measured ones and
        # the fit could only return the identity. Writing that would assert a calibration
        # this scan never measured.
        logger.debug('Positions were not refined; skipping the affine fit.')
        return

    try:
        fit = estimate_affine_transform(
            [(measured, reconstructed)], rng=numpy.random.default_rng(seed)
        )
    except (ValueError, RuntimeError) as exc:
        logger.warning('No affine fit for this scan: %s', exc)
        return

    affine_file = output_directory / 'affine_fit.yaml'
    logger.info('Writing %s', affine_file)
    write_affine_fit(affine_file, fit)


def _build_initial_probe(
    settings: BatchSettings,
    registry: PluginRegistry,
    probe_geometry: ProbeGeometry,
    *,
    photon_wavelength_m: float,
    num_patterns: int,
    rng: numpy.random.Generator,
) -> ProbeSequence:
    """The starting probe: the zone-plate model, or one file shared by every scan."""
    initial = settings.initial_probe

    if initial.probe_source == 'simulated':
        preset = initial.simulated_probe.fzp_preset
        logger.info('Simulating the initial probe from zone plate preset "%s"', preset)
        zone_plate = registry.fresnel_zone_plates.get_strategy_by_name(preset)
        probes = ProbeSequence(
            array=generate_fresnel_zone_plate_probe(
                probe_geometry,
                zone_plate,
                photon_wavelength_m=photon_wavelength_m,
                defocus_distance_m=initial.simulated_probe.fzp_defocus_m,
            ).get_array(),
            opr_weights=None,
            pixel_geometry=probe_geometry.get_pixel_geometry(),
        )
    else:
        file_path = initial.probe_from_file.file_path
        assert file_path is not None  # the settings model requires it for "file"
        logger.info('Reading the initial probe from %s', file_path)
        reader = registry.probe_file_readers.get_strategy_by_name(initial.probe_from_file.file_type)

        # Resampled against *this* scan's geometry rather than once for the run: the
        # probe pitch follows the wavelength and detector distance, and both are resolved
        # per scan. A no-op whenever they agree, and it is what lets a warm start come
        # from a reconstruction done at a different resolution.
        probes = resample_probe_sequence(reader.read(file_path), probe_geometry)

    probe = probes.get_probe_no_opr()

    # Only ever grows the stack. A file carrying more modes than were asked for keeps
    # them, because discarding reconstructed modes is the worse of the two surprises.
    if initial.num_probe_modes > 1 and probe.get_array().shape[0] < initial.num_probe_modes:
        probe = generate_incoherent_probe_modes(probe, initial.num_probe_modes)

    return generate_coherent_probe_modes(
        rng,
        probe,
        num_cmodes=initial.num_opr_modes,
        num_diffraction_patterns=num_patterns,
    )


def _build_initial_object(
    settings: BatchSettings,
    registry: PluginRegistry,
    positions: ProbePositionSequence,
    probe_geometry: ProbeGeometry,
) -> Object:
    """The starting object: transparent, or one file shared by every scan."""
    initial = settings.initial_object
    num_slices = initial.num_slices
    thickness_m = initial.object_thickness_m or 0.0
    layer_spacing_m = compute_uniform_layer_spacing_m(thickness_m, num_slices)
    object_geometry = compute_object_geometry(
        positions, probe_geometry, padding_px=initial.object_padding_px
    )

    if initial.object_source == 'simulated':
        # Unit amplitude and zero phase: a guess that asserts nothing about the sample.
        object_ = generate_uniform_object(object_geometry)

        if num_slices > 1:
            object_ = resize_object_layers(
                object_,
                num_slices,
                fill_mode=LayerFillMode.EDGE,
                layer_spacing_m=layer_spacing_m,
            )

        return object_

    file_path = initial.object_from_file.file_path
    assert file_path is not None  # the settings model requires it for "file"
    logger.info('Reading the initial object from %s', file_path)
    reader = registry.object_file_readers.get_strategy_by_name(initial.object_from_file.file_type)
    source = reader.read(file_path)

    # A file recording its own sampling and center keeps them, so the pixel size can be
    # reconciled below; one recording neither is taken to be sampled the way this scan is.
    try:
        source_pixel_geometry = source.get_pixel_geometry()
    except ValueError:
        source_pixel_geometry = object_geometry.get_pixel_geometry()

    try:
        source_center = source.get_center()
    except ValueError:
        source_center = object_geometry.get_center()

    object_ = Object(
        array=source.get_array(),
        pixel_geometry=source_pixel_geometry,
        center=source_center,
        # The file's own layer count, spread over the depth this run asked for.
        layer_spacing_m=compute_uniform_layer_spacing_m(thickness_m, source.num_layers),
    )
    object_ = resample_object_to_pixel_size(object_, object_geometry.get_pixel_geometry())

    # VACUUM, not EDGE: it is the only fill leaving the layer product unchanged, so a
    # warm start keeps the total transmission it was reconstructed with when that is
    # spread across more slices than it arrived with.
    return resize_object_layers(
        object_,
        num_slices,
        fill_mode=LayerFillMode.VACUUM,
        layer_spacing_m=layer_spacing_m,
    )


def reconstruct_one_scan(
    settings: BatchSettings,
    settings_text: str,
    registry: PluginRegistry,
    request: ScanRequest,
    options: object,
    *,
    read_region: CropRegion | None,
    bad_pixels: BadPixels | None,
    cancellation: CancellationToken,
) -> bool:
    """Reconstruct one projection. Returns False when it was cancelled before finishing."""
    output_directory = settings.output_directory(request.scan_no)
    diffraction_file_path = settings.diffraction_file(request.scan_no)
    position_file_path = settings.position_file(request.scan_no)

    diffraction_reader = registry.diffraction_file_readers.get_strategy_by_name(DIFFRACTION_READER)
    position_reader = registry.probe_position_file_readers.get_strategy_by_name(POSITION_READER)

    logger.info('Reading diffraction from %s', diffraction_file_path)
    raw_dataset = diffraction_reader.read(diffraction_file_path)
    metadata = raw_dataset.get_metadata()
    overrides = settings.metadata_overrides

    detector_distance_m = resolve_quantity(
        logger,
        'Detector distance (m)',
        overrides.detector_distance_m,
        metadata.detector_distance_m,
        None,
        'metadata_overrides.detector_distance_m',
        level=logging.DEBUG,
    )
    photon_energy_eV = resolve_quantity(  # noqa: N806
        logger,
        'Photon energy (eV)',
        overrides.probe_energy_eV,
        metadata.photon_energy_eV,
        None,
        'metadata_overrides.probe_energy_eV',
        level=logging.DEBUG,
    )

    if metadata.detector_pixel_geometry is None:
        raise ValueError('The diffraction file records no detector pixel size!')

    detector_pixel_size_m = metadata.detector_pixel_geometry.width_m
    logger.debug('Detector pixel size (m): %g (from the diffraction file)', detector_pixel_size_m)
    raw_pixel_geometry = PixelGeometry(
        width_m=detector_pixel_size_m, height_m=detector_pixel_size_m
    )

    max_valid_count = (
        invalid_count_threshold(metadata.pattern_dtype)
        if settings.diffraction.max_valid_count is None
        else settings.diffraction.max_valid_count
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
        total_counts_lower_bound=settings.diffraction.min_total_counts,
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
        logger, output_directory, assembled_data, skip=not settings.misc.save_diffraction
    )

    measured_positions = position_reader.read(position_file_path)
    logger.info('Read %d probe positions from %s', len(measured_positions), position_file_path)

    transform = settings.positions.get_transform()

    if transform is None:
        positions = measured_positions
    else:
        from ptychodus.api.affine import transform_probe_positions

        positions = ProbePositionSequence(
            [*transform_probe_positions(measured_positions, transform)]
        )

    photon_wavelength_m = energy_eV_to_wavelength_m(photon_energy_eV)
    probe_geometry = ProbeGeometry.from_far_field(
        assembled_data.get_pixel_geometry(),
        assembled_data.get_image_extent(),
        wavelength_m=photon_wavelength_m,
        distance_m=detector_distance_m,
    )

    rng = numpy.random.default_rng(settings.misc.seed)
    probe_sequence = _build_initial_probe(
        settings,
        registry,
        probe_geometry,
        photon_wavelength_m=photon_wavelength_m,
        num_patterns=num_patterns,
        rng=rng,
    )
    object_ = _build_initial_object(settings, registry, positions, probe_geometry)
    logger.info('Object geometry: %d layer(s)', object_.num_layers)

    probe_photon_count = (
        float(assembled_data.get_probe_photon_count())
        if overrides.probe_photon_count is None
        else overrides.probe_photon_count
    )
    tomography_angle_deg = (
        request.encoder_angle_deg
        if overrides.tomography_angle_deg is None
        else overrides.tomography_angle_deg
    )

    product = Product(
        metadata=ProductMetadata(
            name=f'scan{request.scan_no:05d}_{request.label}',
            comments=request.text,
            detector_distance_m=detector_distance_m,
            photon_energy_eV=photon_energy_eV,
            probe_photon_count=probe_photon_count,
            exposure_time_s=float(metadata.exposure_time_s or 0.0),
            mass_attenuation_m2_per_kg=0.0,
            tomography_angle_deg=tomography_angle_deg,
        ),
        probe_positions=positions,
        probes=probe_sequence,
        object_=object_,
        losses=[],
    )

    reconstruct_input = prepare_reconstruct_input(assembled_data, product)

    if settings.misc.save_input_product:
        input_file = output_directory / 'input_product.h5'
        logger.info('Writing %s', input_file)
        save_product(input_file, reconstruct_input.product)

    settings_file = output_directory / 'batch_settings.yaml'
    logger.info('Writing %s', settings_file)
    settings_file.write_text(settings_text)

    final_product = run_reconstruction(
        logger,
        reconstruct_input,
        options,  # type: ignore[arg-type]
        output_directory,
        num_sync_epochs=settings.misc.num_sync_epochs,
        cancellation=cancellation,
    )

    if final_product is None:
        return False

    _estimate_and_write_affine_fit(
        output_directory,
        # The measured positions as the reconstruction saw them: pattern-paired, so they
        # index-match the refined positions coming back out.
        reconstruct_input.product.probe_positions,
        final_product.probe_positions,
        seed=settings.misc.seed,
    )
    return True


def product_exists(settings: BatchSettings, scan_no: int) -> bool:
    """Whether this scan already has a final product, for a resumed run."""
    return StandardFileLayout.PRODUCT.path(settings.output_directory(scan_no)).is_file()

#!/usr/bin/env python
"""Repackage prepared datasets into ptychodus formats through the ptychodus api.

Two independent conversions, either or both per run. A raw diffraction file is read,
assembled, and written as the standard assembled-pattern HDF5. A data product is either
read from a product file or built from the assembled patterns and a probe position file,
and is written out through a product writer plugin.

fold_slice is handled as a special case, because it is what this is mostly pointed at.
Its preprocessing step writes each scan as a pair that differs only by suffix --
``<stem>_dp.hdf5`` holds the patterns, ``<stem>_para.hdf5`` the probe positions -- and
neither file points at the other, so the pairing is the filename and this derives it.
The parameter file also records the probe wavelength, the sample-plane pixel size and
the tomography angle, which between them give the probe energy and the detector
distance, so a measured scan converts with no geometry on the command line at all:

convert-to-ptychodus \
    --diffraction-input  "data/IC_1/data_roi0_Ndp256_rs1_dp.hdf5" \
    --diffraction-output "data/IC_1/diffraction.h5" \
    --product-output     "data/IC_1/product.h5"

A simulated scan carries only the coordinates, so there --detector-distance-m and
--probe-energy-eV have to be given. Either one also overrides what the file recorded.

The other fold_slice path repackages a finished reconstruction. That format records no
detector distance, probe photon count or exposure time, so the same metadata arguments
fill them in over whatever the file supplied:

convert-to-ptychodus \
    --product-input      "data/IC_1/Niter100.mat" \
    --product-input-type "fold_slice_mat" \
    --product-output     "data/IC_1/product.h5" \
    --detector-distance-m 2.335

Run with --list-plugins to see the reader and writer names every --*-type argument
accepts. An unrecognized name is an error naming the registered plugins, so a
conversion never silently proceeds with a reader other than the one asked for.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import argparse
import json
import logging
import sys

import numpy

from ptychodus.api.assemble import AssembledDiffractionData, assemble_dataset
from ptychodus.api.constants import energy_eV_to_wavelength_m, wavelength_m_to_energy_eV
from ptychodus.api.diffraction import BadPixels
from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.io import save_diffraction_data
from ptychodus.api.object import Object, ObjectCenter, ObjectGeometry, compute_object_geometry
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe import Probe, ProbeGeometry, ProbeSequence
from ptychodus.api.probe_positions import ProbePositionSequence
from ptychodus.api.product import Product, ProductMetadata
from ptychodus.api.propagate import compute_far_field_propagation_distance
from ptychodus.api.simulate.object import generate_uniform_object
from ptychodus.api.simulate.probe import (
    generate_average_pattern_probe,
    generate_coherent_probe_modes,
    generate_fresnel_zone_plate_probe,
    generate_incoherent_probe_modes,
)
from ptychodus.cli import positive_int

# Imported directly rather than through the registry because no plugin interface carries
# either one: a ProbePositionFileReader returns coordinates, so the rest of what the
# parameter file records has nowhere to go, and a filename convention is not a plugin at
# all. reconstruct_batch_lamni reaches into its instrument's plugin the same way.
from ptychodus.plugins.fold_slice._pairing import find_position_file
from ptychodus.plugins.fold_slice.position_file import (
    FoldSliceParameters,
    read_fold_slice_parameters,
)
import ptychodus

logger = logging.getLogger(__name__)

# Which plugins read the fold_slice pair. The filename convention that pairs the two
# files is the format's own, so it lives with the format in `plugins.fold_slice`.
FOLD_SLICE_DIFFRACTION_READER = 'fold_slice'
FOLD_SLICE_POSITION_READER = 'fold_slice'


def _resolve_raw_pixel_geometry(
    override_m: float | None, from_file: PixelGeometry | None
) -> PixelGeometry:
    """Pick the raw detector pixel pitch from the command line or the file, and say which.

    Reporting the source is the point: the pitch sets the object pixel size everything
    downstream is expressed in, and a run that substituted one value for another would
    otherwise be indistinguishable from one that read it.
    """
    if override_m is not None:
        logger.info('Detector pixel size (m): %g (from --detector-pixel-size-m)', override_m)
        return PixelGeometry(width_m=override_m, height_m=override_m)

    if from_file is not None:
        logger.info(
            'Detector pixel size (m): %g x %g (from the diffraction file)',
            from_file.width_m,
            from_file.height_m,
        )
        return from_file

    raise ValueError(
        'The diffraction file records no detector pixel geometry; pass --detector-pixel-size-m.'
    )


def _read_bad_pixels(
    registry: PluginRegistry, file_path: Path | None, file_type: str
) -> BadPixels | None:
    if file_path is None:
        return None

    logger.info('Reading bad pixels from %s as %s', file_path, file_type)
    reader = registry.bad_pixels_file_readers.get_strategy_by_name(file_type)
    return reader.read(file_path)


def _assemble_diffraction(
    registry: PluginRegistry, args: argparse.Namespace
) -> AssembledDiffractionData:
    """Read a raw diffraction file and assemble its patterns."""
    logger.info('Reading %s as %s', args.diffraction_input, args.diffraction_input_type)
    reader = registry.diffraction_file_readers.get_strategy_by_name(args.diffraction_input_type)
    dataset = reader.read(args.diffraction_input)
    metadata = dataset.get_metadata()

    num_expected_patterns = sum(metadata.num_patterns_per_array)
    logger.info(
        'Series: %d array(s), %d pattern(s), detector %dx%d',
        len(metadata.num_patterns_per_array),
        num_expected_patterns,
        metadata.detector_extent.width_px,
        metadata.detector_extent.height_px,
    )

    raw_pixel_geometry = _resolve_raw_pixel_geometry(
        args.detector_pixel_size_m, metadata.detector_pixel_geometry
    )
    bad_pixels = _read_bad_pixels(registry, args.bad_pixels_file, args.bad_pixels_file_type)

    # No preprocessing pipeline: this repackages the patterns as recorded, leaving the
    # crop, binning and flips to whoever reconstructs them.
    assembled_data = assemble_dataset(
        dataset,
        bad_pixels=bad_pixels,
        raw_pixel_geometry=raw_pixel_geometry,
        total_counts_lower_bound=args.min_total_counts,
        total_counts_upper_bound=args.max_total_counts,
    )
    num_patterns = assembled_data.get_num_patterns()

    if num_patterns != num_expected_patterns:
        logger.warning(
            'Assembled %d of %d patterns; %d were filtered or failed to load.',
            num_patterns,
            num_expected_patterns,
            num_expected_patterns - num_patterns,
        )

    return assembled_data


def _resolve_position_file(args: argparse.Namespace) -> tuple[Path, str] | None:
    """Locate the probe positions to build a product from, as a (path, plugin name) pair.

    An explicit --probe-positions wins. Failing that, a fold_slice pattern file names its
    own companion, so the pair needs no second argument on the command line; the
    companion has to exist, since a derived name that happens to be wrong should fall
    through to the "nothing to build from" error rather than to a read failure.
    """
    if args.probe_positions is not None:
        return args.probe_positions, args.probe_positions_type

    if args.diffraction_input is None:
        return None

    if args.diffraction_input_type != FOLD_SLICE_DIFFRACTION_READER:
        return None

    companion = find_position_file(args.diffraction_input)

    if companion is None or not companion.is_file():
        return None

    return companion, FOLD_SLICE_POSITION_READER


def _apply_metadata_overrides(
    metadata: ProductMetadata, args: argparse.Namespace
) -> ProductMetadata:
    """Apply every metadata argument that was given, leaving the rest as they arrived.

    These are what makes a fold_slice product usable at all: the formats record no
    detector distance, probe photon count or exposure time, so a product read from one
    carries 0.0 for each, and a zero detector distance collapses the sample-plane pixel
    size to zero wherever it is derived.
    """
    overrides = {
        'name': args.product_name,
        'detector_distance_m': args.detector_distance_m,
        'photon_energy_eV': args.photon_energy_eV,
        'probe_photon_count': args.probe_photon_count,
        'exposure_time_s': args.exposure_time_s,
        'mass_attenuation_m2_per_kg': args.mass_attenuation_m2_per_kg,
        'tomography_angle_deg': args.tomography_angle_deg,
    }
    given = {key: value for key, value in overrides.items() if value is not None}
    return replace(metadata, **given) if given else metadata


def _inherit_object_geometry(
    override: Object, *, pixel_geometry: PixelGeometry | None, center: ObjectCenter | None
) -> Object:
    """Give an object read from file the sampling and placement it does not record itself.

    No object format records a pixel size or a center, so one read from file arrives with
    neither and `save_product` refuses to write it. Both fall back to the geometry of the
    product the object is going into.
    """
    override_pixel_geometry = _object_pixel_geometry(override)
    override_center = _object_center(override)

    return Object(
        array=override.get_array(),
        pixel_geometry=(
            pixel_geometry if override_pixel_geometry is None else override_pixel_geometry
        ),
        center=center if override_center is None else override_center,
        layer_spacing_m=override.layer_spacing_m,
    )


def _object_pixel_geometry(object_: Object) -> PixelGeometry | None:
    try:
        return object_.get_pixel_geometry()
    except ValueError:
        return None


def _object_center(object_: Object) -> ObjectCenter | None:
    try:
        return object_.get_center()
    except ValueError:
        return None


def _inherit_probe_geometry(
    override: ProbeSequence, *, pixel_geometry: PixelGeometry | None
) -> ProbeSequence:
    """Give a probe read from file the sampling it does not record itself.

    The counterpart of :func:`_inherit_object_geometry`; no probe format records a pixel
    size either.
    """
    override_pixel_geometry = _probe_pixel_geometry(override)

    return ProbeSequence(
        array=override.get_array(),
        opr_weights=override.get_opr_weights_or_none(),
        pixel_geometry=(
            pixel_geometry if override_pixel_geometry is None else override_pixel_geometry
        ),
    )


def _probe_pixel_geometry(probes: ProbeSequence) -> PixelGeometry | None:
    try:
        return probes.get_pixel_geometry()
    except ValueError:
        return None


@dataclass(frozen=True)
class _BuiltProductGeometry:
    """The three values a built product cannot derive from the patterns and the scan."""

    detector_distance_m: float
    photon_energy_eV: float  # noqa: N815
    tomography_angle_deg: float


def _resolve_built_product_geometry(
    args: argparse.Namespace,
    parameters: FoldSliceParameters,
    assembled_data: AssembledDiffractionData,
) -> _BuiltProductGeometry:
    """Settle the probe energy, detector distance and tomography angle, and say from where.

    Precedence is the command line, then the parameter file, and nothing after that: a
    guessed detector distance or probe energy would reconstruct at the wrong scale with
    no sign of it in the output, so an unrecorded value is an error naming the argument
    that supplies it. Reporting the source is what distinguishes a run that read the
    geometry from one that was told it.
    """
    photon_energy_eV = args.photon_energy_eV  # noqa: N806

    if photon_energy_eV is not None:
        logger.info('Photon energy (eV): %g (from --photon-energy-eV)', photon_energy_eV)
    elif parameters.photon_wavelength_m is not None:
        photon_energy_eV = wavelength_m_to_energy_eV(parameters.photon_wavelength_m)  # noqa: N806
        logger.info(
            'Photon energy (eV): %g (from the %g m wavelength in the parameter file)',
            photon_energy_eV,
            parameters.photon_wavelength_m,
        )
    else:
        raise ValueError('The parameter file records no probe wavelength; pass --probe-energy-eV.')

    detector_distance_m = args.detector_distance_m

    if detector_distance_m is not None:
        logger.info('Detector distance (m): %g (from --detector-distance-m)', detector_distance_m)
    elif parameters.object_pixel_size_m is not None:
        detector_distance_m = compute_far_field_propagation_distance(
            assembled_data.get_pixel_geometry(),
            assembled_data.get_image_extent(),
            wavelength_m=energy_eV_to_wavelength_m(photon_energy_eV),
            conjugate_pixel_width_m=parameters.object_pixel_size_m,
        )
        logger.info(
            'Detector distance (m): %g (from the %g m object pixel size in the parameter file)',
            detector_distance_m,
            parameters.object_pixel_size_m,
        )
    else:
        raise ValueError(
            'The parameter file records no object pixel size; pass --detector-distance-m.'
        )

    tomography_angle_deg = args.tomography_angle_deg

    if tomography_angle_deg is None:
        tomography_angle_deg = (
            0.0 if parameters.tomography_angle_deg is None else parameters.tomography_angle_deg
        )

    return _BuiltProductGeometry(
        detector_distance_m=detector_distance_m,
        photon_energy_eV=photon_energy_eV,
        tomography_angle_deg=tomography_angle_deg,
    )


def _read_fold_slice_parameters_if_available(
    file_path: Path, file_type: str
) -> FoldSliceParameters:
    """The geometry beside the positions, or nothing for a format that records none."""
    if file_type != FOLD_SLICE_POSITION_READER:
        return FoldSliceParameters()

    return read_fold_slice_parameters(file_path)


def _build_probe(
    registry: PluginRegistry,
    args: argparse.Namespace,
    assembled_data: AssembledDiffractionData,
    geometry: _BuiltProductGeometry,
    probe_geometry: ProbeGeometry,
    photon_wavelength_m: float,
    rng: numpy.random.Generator,
) -> ProbeSequence:
    """Supply the initial probe for a built product, from file or from a model."""
    if args.override_probe is not None:
        logger.info('Reading probe from %s as %s', args.override_probe, args.override_probe_type)
        reader = registry.probe_file_readers.get_strategy_by_name(args.override_probe_type)
        array = reader.read(args.override_probe).get_probe_no_opr().get_array()

        # A probe sits on the grid the patterns define; a mismatch means it was written
        # against a different crop, and nothing downstream would reveal it.
        if array.shape[-2:] != (probe_geometry.height_px, probe_geometry.width_px):
            raise ValueError(
                f'Probe file extent {array.shape[-2]}x{array.shape[-1]} does not match the '
                f'patterns {probe_geometry.height_px}x{probe_geometry.width_px}.'
            )

        probe = Probe(array=array, pixel_geometry=probe_geometry.get_pixel_geometry())
    elif args.fzp_preset:
        logger.info('Simulating the initial probe from zone plate preset "%s"', args.fzp_preset)
        zone_plate = registry.fresnel_zone_plates.get_strategy_by_name(args.fzp_preset)
        probe = generate_fresnel_zone_plate_probe(
            probe_geometry,
            zone_plate,
            photon_wavelength_m=photon_wavelength_m,
            defocus_distance_m=args.fzp_defocus_m,
        )
    else:
        # fold_slice is written by several instruments, so no one zone plate applies.
        # Back-propagating the mean pattern estimates the probe without asserting an
        # optic at all, which serves zone plates, KB mirrors and pinholes alike.
        logger.info('Estimating the initial probe by back-propagating the mean pattern')
        probe = generate_average_pattern_probe(
            probe_geometry,
            assembled_data,
            photon_wavelength_m=photon_wavelength_m,
            detector_distance_m=geometry.detector_distance_m,
        )

    if args.num_probe_modes > 1:
        probe = generate_incoherent_probe_modes(probe, args.num_probe_modes)

    return generate_coherent_probe_modes(
        rng,
        probe,
        num_cmodes=args.num_opr_modes,
        num_diffraction_patterns=assembled_data.get_num_patterns(),
    )


def _build_object(
    registry: PluginRegistry, args: argparse.Namespace, object_geometry: ObjectGeometry
) -> Object:
    """Supply the initial object for a built product, from file or as a flat canvas."""
    if args.override_object is not None:
        logger.info('Reading object from %s as %s', args.override_object, args.override_object_type)
        reader = registry.object_file_readers.get_strategy_by_name(args.override_object_type)
        return _inherit_object_geometry(
            reader.read(args.override_object),
            pixel_geometry=object_geometry.get_pixel_geometry(),
            center=object_geometry.get_center(),
        )

    # Deviations of zero give a flat unit-amplitude, zero-phase field.
    return generate_uniform_object(object_geometry)


def _build_product(
    registry: PluginRegistry,
    args: argparse.Namespace,
    assembled_data: AssembledDiffractionData,
    positions: ProbePositionSequence,
    geometry: _BuiltProductGeometry,
) -> Product:
    """Construct a product from assembled patterns, probe positions and their geometry.

    Everything `geometry` does not carry is derived from what the patterns and the scan
    already say: the probe grid from the Fraunhofer relation, the object canvas from the
    scan bounding box, the photon count from the brightest pattern.
    """
    photon_wavelength_m = energy_eV_to_wavelength_m(geometry.photon_energy_eV)
    probe_geometry = ProbeGeometry.from_far_field(
        assembled_data.get_pixel_geometry(),
        assembled_data.get_image_extent(),
        wavelength_m=photon_wavelength_m,
        distance_m=geometry.detector_distance_m,
    )
    logger.info('Probe geometry: %s', probe_geometry)

    rng = numpy.random.default_rng(args.seed)
    probes = _build_probe(
        registry, args, assembled_data, geometry, probe_geometry, photon_wavelength_m, rng
    )
    logger.info('Probe: %s', probes.get_array().shape)

    object_geometry = compute_object_geometry(
        positions, probe_geometry, padding_px=args.object_padding_px
    )
    logger.info('Object geometry: %s', object_geometry)
    object_ = _build_object(registry, args, object_geometry)

    probe_photon_count = (
        float(assembled_data.get_probe_photon_count())
        if args.probe_photon_count is None
        else float(args.probe_photon_count)
    )

    return Product(
        metadata=ProductMetadata(
            name=_built_product_name(args),
            comments=f'Converted from {args.diffraction_input.name}',
            detector_distance_m=geometry.detector_distance_m,
            photon_energy_eV=geometry.photon_energy_eV,
            probe_photon_count=probe_photon_count,
            exposure_time_s=0.0 if args.exposure_time_s is None else args.exposure_time_s,
            mass_attenuation_m2_per_kg=(
                0.0 if args.mass_attenuation_m2_per_kg is None else args.mass_attenuation_m2_per_kg
            ),
            tomography_angle_deg=geometry.tomography_angle_deg,
        ),
        probe_positions=positions,
        probes=probes,
        object_=object_,
        losses=[],
    )


def _built_product_name(args: argparse.Namespace) -> str:
    """Name a built product after --product-name, else its scan directory.

    The directory rather than the file: fold_slice names every scan's patterns with the
    same generic "data_roi0..._dp" form, so the stem identifies the preprocessing run
    and the directory identifies the scan.
    """
    if args.product_name is not None:
        return args.product_name

    directory = args.diffraction_input.parent.name
    return directory if directory else args.diffraction_input.stem


def _apply_product_overrides(
    registry: PluginRegistry, product: Product, args: argparse.Namespace
) -> Product:
    """Fold the metadata and component arguments onto a product read from file."""
    metadata = _apply_metadata_overrides(product.metadata, args)

    if metadata is not product.metadata:
        product = replace(product, metadata=metadata)

    if args.override_object is not None:
        logger.info('Reading object from %s as %s', args.override_object, args.override_object_type)
        reader = registry.object_file_readers.get_strategy_by_name(args.override_object_type)
        object_ = _inherit_object_geometry(
            reader.read(args.override_object),
            pixel_geometry=_object_pixel_geometry(product.object_),
            center=_object_center(product.object_),
        )
        product = replace(product, object_=object_)

    if args.override_probe is not None:
        logger.info('Reading probe from %s as %s', args.override_probe, args.override_probe_type)
        probe_reader = registry.probe_file_readers.get_strategy_by_name(args.override_probe_type)
        probes = _inherit_probe_geometry(
            probe_reader.read(args.override_probe),
            pixel_geometry=_probe_pixel_geometry(product.probes),
        )
        product = replace(product, probes=probes)

    if args.probe_positions is not None:
        logger.info(
            'Reading probe positions from %s as %s',
            args.probe_positions,
            args.probe_positions_type,
        )
        position_reader = registry.probe_position_file_readers.get_strategy_by_name(
            args.probe_positions_type
        )
        product = replace(product, probe_positions=position_reader.read(args.probe_positions))

    return product


def _read_positions(
    registry: PluginRegistry, file_path: Path, file_type: str
) -> ProbePositionSequence:
    logger.info('Reading probe positions from %s as %s', file_path, file_type)
    reader = registry.probe_position_file_readers.get_strategy_by_name(file_type)
    positions = reader.read(file_path)
    logger.info('Read %d probe positions', len(positions))
    return positions


def _list_plugins(registry: PluginRegistry) -> None:
    plugins: dict[str, str] = {
        'diffraction_readers': registry.diffraction_file_readers.stringify_plugin_names(),
        'product_readers': registry.product_file_readers.stringify_plugin_names(),
        'product_writers': registry.product_file_writers.stringify_plugin_names(),
        'probe_position_readers': registry.probe_position_file_readers.stringify_plugin_names(),
        'probe_readers': registry.probe_file_readers.stringify_plugin_names(),
        'object_readers': registry.object_file_readers.stringify_plugin_names(),
        'bad_pixels_readers': registry.bad_pixels_file_readers.stringify_plugin_names(),
        'fresnel_zone_plates': registry.fresnel_zone_plates.stringify_plugin_names(),
    }

    print(json.dumps(plugins, indent=4))


def _create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='convert-to-ptychodus repackages prepared datasets into ptychodus formats.',
    )
    parser.add_argument(
        '--diffraction-input-type',
        default=FOLD_SLICE_DIFFRACTION_READER,
        help='Diffraction input file type.',
    )
    parser.add_argument(
        '--diffraction-input',
        metavar='DIFFRACTION_INPUT_FILE',
        type=Path,
        help='Path to the diffraction input file.',
    )
    parser.add_argument(
        '--diffraction-output',
        metavar='DIFFRACTION_OUTPUT_FILE',
        type=Path,
        help='Path to the diffraction output file.',
    )
    parser.add_argument(
        '--detector-pixel-size-m',
        type=float,
        default=None,
        help='Detector pixel pitch, both axes. Overrides the diffraction file.',
    )
    parser.add_argument(
        '--min-total-counts',
        type=int,
        default=None,
        help='Drop patterns whose total counts over the good pixels fall below this.',
    )
    parser.add_argument(
        '--max-total-counts',
        type=int,
        default=None,
        help='Drop patterns whose total counts over the good pixels exceed this.',
    )
    parser.add_argument(
        '--bad-pixels-file',
        metavar='BAD_PIXELS_FILE',
        type=Path,
        default=None,
        help='Bad-pixel mask covering the full detector. Overrides the diffraction file.',
    )
    parser.add_argument(
        '--bad-pixels-file-type',
        default='NPY_Bad_Pixels',
        help='Bad pixels file type.',
    )
    parser.add_argument(
        '--list-plugins',
        action='store_true',
        help='List available file reader plugins, then exit.',
    )
    parser.add_argument(
        '--log-level',
        type=int,
        default=logging.INFO,
        help='Set Python logging level.',
    )
    parser.add_argument(
        '--override-object-type',
        metavar='OBJECT_FILE_TYPE',
        default=FOLD_SLICE_DIFFRACTION_READER,
        help='Override object file type.',
    )
    parser.add_argument(
        '--override-object',
        metavar='OBJECT_FILE',
        type=Path,
        help='Path to the object file.',
    )
    parser.add_argument(
        '--override-probe-type',
        metavar='PROBE_FILE_TYPE',
        default=FOLD_SLICE_DIFFRACTION_READER,
        help='Override probe file type.',
    )
    parser.add_argument(
        '--override-probe',
        metavar='PROBE_FILE',
        type=Path,
        help='Path to the probe file.',
    )
    parser.add_argument(
        '--probe-positions-type',
        metavar='PROBE_POSITIONS_FILE_TYPE',
        default=FOLD_SLICE_POSITION_READER,
        help='Probe positions file type.',
    )
    parser.add_argument(
        '--probe-positions',
        metavar='PROBE_POSITIONS_FILE',
        type=Path,
        help=(
            'Path to the probe positions file. For a fold_slice "_dp" diffraction input '
            'the matching "_para" file is found automatically.'
        ),
    )
    parser.add_argument(
        '--product-input-type',
        default=FOLD_SLICE_DIFFRACTION_READER,
        help='Product input file type.',
    )
    parser.add_argument(
        '--product-input',
        metavar='PRODUCT_INPUT_FILE',
        type=Path,
        help='Path to the product input file. Omit to build a product from the diffraction.',
    )
    parser.add_argument(
        '--product-name',
        help='Data product name. A built product defaults to its scan directory name.',
    )
    parser.add_argument(
        '--product-output',
        metavar='PRODUCT_OUTPUT_FILE',
        type=Path,
        help='Path to the product output file.',
    )
    parser.add_argument(
        '--product-output-type',
        default='HDF5',
        help='Product output file type.',
    )
    parser.add_argument(
        '--detector-distance-m',
        type=float,
        default=None,
        help=(
            'Sample-to-detector distance. Derived from the fold_slice object pixel size '
            'when the parameter file records one.'
        ),
    )
    parser.add_argument(
        '--photon-energy-eV',
        type=float,
        default=None,
        help=(
            'Photon energy in electron volts. Derived from the fold_slice wavelength when '
            'the parameter file records one.'
        ),
    )
    parser.add_argument(
        '--probe-photon-count',
        type=float,
        default=None,
        help='Per-snapshot probe photon count. Defaults to the brightest pattern.',
    )
    parser.add_argument(
        '--exposure-time-s',
        type=float,
        default=None,
        help='Exposure time in seconds.',
    )
    parser.add_argument(
        '--mass-attenuation-m2-kg',
        type=float,
        default=None,
        help='Mass attenuation coefficient.',
    )
    parser.add_argument(
        '--tomography-angle-deg',
        type=float,
        default=None,
        help='Tomography angle in degrees. Defaults to the fold_slice parameter file.',
    )
    parser.add_argument(
        '--fzp-preset',
        default='',
        help=(
            'FresnelZonePlate plugin preset for a generated probe. fold_slice is written by '
            'several instruments, so no one zone plate applies and this is empty by default: '
            'the probe is estimated from the data instead.'
        ),
    )
    parser.add_argument(
        '--fzp-defocus-m',
        type=float,
        default=0.0,
        help='Defocus from the zone-plate focal plane. Ignored without --fzp-preset.',
    )
    parser.add_argument(
        '--num-probe-modes',
        type=positive_int,
        default=1,
        help='Incoherent probe modes in a generated probe.',
    )
    parser.add_argument(
        '--num-opr-modes',
        type=positive_int,
        default=1,
        help='Coherent (OPR) probe modes in a generated probe; 1 disables OPR.',
    )
    parser.add_argument(
        '--object-padding-px',
        type=int,
        default=64,
        help='Object-canvas margin added to each side of the scan bounding box.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='RNG seed for generated probes and objects.',
    )
    parser.add_argument(
        '-v',
        '--version',
        action='version',
        version=ptychodus.VERSION_STRING,
    )
    return parser


def main() -> int:
    parser = _create_argument_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        stream=sys.stderr,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )

    registry = PluginRegistry.load_plugins()

    if args.list_plugins:
        _list_plugins(registry)
        return 0

    position_source = _resolve_position_file(args)
    build_product = args.product_output is not None and args.product_input is None

    # Each conversion is validated against what it needs, so that a product-only run and
    # a diffraction-only run are both possible and neither demands the other's arguments.
    if args.diffraction_output is not None and args.diffraction_input is None:
        parser.error('--diffraction-output requires --diffraction-input')

    if build_product:
        if args.diffraction_input is None:
            parser.error(
                '--product-output without --product-input builds a product, '
                'which requires --diffraction-input'
            )

        if position_source is None:
            parser.error(
                'building a product requires probe positions: pass --probe-positions, or '
                'name a fold_slice "_dp" diffraction input whose "_para" file sits beside it'
            )

    if args.diffraction_output is None and args.product_output is None:
        parser.error('nothing to write: pass --diffraction-output and/or --product-output')

    assembled_data = (
        None if args.diffraction_input is None else _assemble_diffraction(registry, args)
    )

    if args.diffraction_output is not None:
        assert assembled_data is not None
        save_diffraction_data(args.diffraction_output, assembled_data)
        logger.info('Wrote diffraction data to %s', args.diffraction_output)

    if args.product_output is None:
        return 0

    if build_product:
        assert assembled_data is not None and position_source is not None
        position_file, position_type = position_source
        parameters = _read_fold_slice_parameters_if_available(position_file, position_type)
        geometry = _resolve_built_product_geometry(args, parameters, assembled_data)
        positions = _read_positions(registry, position_file, position_type)
        product = _build_product(registry, args, assembled_data, positions, geometry)
    else:
        logger.info('Reading %s as %s', args.product_input, args.product_input_type)
        reader = registry.product_file_readers.get_strategy_by_name(args.product_input_type)
        product = _apply_product_overrides(registry, reader.read(args.product_input), args)

    writer = registry.product_file_writers.get_strategy_by_name(args.product_output_type)
    writer.write(args.product_output, product)
    logger.info('Wrote product data to %s', args.product_output)

    return 0


if __name__ == '__main__':
    sys.exit(main())

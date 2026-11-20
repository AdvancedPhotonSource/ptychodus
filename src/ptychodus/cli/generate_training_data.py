#!/usr/bin/env python
"""Write training data for a learned reconstructor, measured or synthesized.

By default the command exports a measured scan: the patterns of a prepared directory, or of
a storage-root product named by uuid, paired with the product's positions by scan index the
way a reconstruction pairs them, written in a training-data format. With ``--synthetic``
it reads only the product and keeps its probe, positions and metadata, but replaces the
object with one drawn from an object model that needs no measured data -- ``dead_leaves``
gives natural-image statistics -- and simulates the patterns, with Poisson noise at the
product's photon scale and an ideal detector. ``--count`` draws that many independent
objects, and ``--position-jitter-m`` perturbs the positions of each, recording the
perturbed positions the patterns were simulated at.

Every file is numbered ``<stem>_NNNN`` after the highest index already in the output
directory, so measured and synthetic sets can be added to one directory over several runs,
and ``ptychodus-train`` reads it as one training set. Nothing is ever overwritten: an index
is first reserved by creating the hidden directory ``.staging-<stem>_NNNN``, which only one
run can create, then the writer writes there and its files are linked into place by a call
that fails rather than replaces. Concurrent runs into one directory therefore never share an
index. A synthetic realization's random stream is derived from ``--seed`` and its index, so
a file can be reproduced on its own and a continued run with the same seed adds new data.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import UUID
import argparse
import logging
import math
import os
import re
import shutil
import sys

import numpy
from pydantic import BaseModel

import ptychodus
from ptychodus.api.assemble import AssembledDiffractionData
from ptychodus.api.accumulate import DiffractionMoments
from ptychodus.api.exit_codes import ExitCode
from ptychodus.api.io import StandardFileLayout, load_diffraction_data, load_product
from ptychodus.api.object import ObjectGeometry, ObjectGeometryProvider, compute_object_geometry
from ptychodus.api.plugins import Plugin, PluginCatalog, PluginRegistry
from ptychodus.api.probe_positions import ProbePosition, ProbePositionSequence
from ptychodus.api.product import Product, ProductMetadata
from ptychodus.api.propagate import compute_product_geometry
from ptychodus.api.reconstruct import (
    PositionIndexFilter,
    ReconstructInput,
    TrainingDataFileWriter,
    prepare_reconstruct_input,
)
from ptychodus.api.simulate.diffraction import generate_diffraction_data
from ptychodus.api.simulate.object import ObjectGenerator
from ptychodus.api.storage import (
    STORAGE_ROOT_ENVIRONMENT_VARIABLE,
    LayoutError,
    ManifestLoadError,
    StorageLayout,
    resolve_storage_inputs,
)
from ptychodus.cli import DirectoryType, add_log_level_argument, configure_logging, positive_int
from ptychodus.api.options import OptionsError, apply_options_document, read_options_document
from ptychodus.cli._options import format_options_document

logger = logging.getLogger('generate_training_data')


class _CanvasGeometryProvider(ObjectGeometryProvider):
    """The geometry an object model samples a synthetic object on, with no measured data.

    The generated object is not scaled to a measured transmission, so the mean amplitude
    transmission is one.
    """

    def __init__(
        self,
        metadata: ProductMetadata,
        positions: Sequence[ProbePosition],
        geometry: ObjectGeometry,
    ) -> None:
        self._metadata = metadata
        self._positions = positions
        self._geometry = geometry

    @property
    def photon_wavelength_m(self) -> float:
        return self._metadata.photon_wavelength_m

    @property
    def mean_amplitude_transmission(self) -> float:
        return 1.0

    @property
    def exit_wave_propagation_distance_m(self) -> float:
        if self._metadata.far_field:
            return self._metadata.detector_distance_m

        geometry = compute_product_geometry(
            photon_energy_eV=self._metadata.photon_energy_eV,
            probe_photon_count=self._metadata.probe_photon_count,
            exposure_time_s=self._metadata.exposure_time_s,
            detector_distance_m=self._metadata.detector_distance_m,
            focus_object_distance_m=self._metadata.focus_object_distance_m,
            far_field=self._metadata.far_field,
        )
        return geometry.object_plane_propagation_distance_m

    @property
    def far_field(self) -> bool:
        return self._metadata.far_field

    def get_probe_positions(self) -> Sequence[ProbePosition]:
        return self._positions

    def get_object_geometry(self) -> ObjectGeometry:
        return self._geometry

    def get_assembled_data(self) -> AssembledDiffractionData | None:
        return None

    def get_diffraction_moments(self) -> DiffractionMoments | None:
        return None


class _TrainingDirectory:
    """Numbered training files under one stem in one directory, never overwritten.

    An index is reserved by creating its staging directory, which fails for every run but
    one, and committed by linking each file the writer produced into place, which fails on
    an existing name rather than replacing it.
    """

    def __init__(self, directory: Path, stem: str) -> None:
        self._directory = directory
        self._stem = stem
        self._numbered = re.compile(rf'^(?:\.staging-)?{re.escape(stem)}_(\d+)')

    def name(self, index: int) -> str:
        return f'{self._stem}_{index:04d}'

    def warn_about_staging(self) -> None:
        for staging in sorted(self._directory.glob('.staging-*')):
            logger.warning(
                f'{staging} is a reservation held by a running export, or left behind by one '
                'that was killed; delete it if no export is running.'
            )

    def reserve(self) -> tuple[int, Path]:
        """Reserve the next free index, returning it with its staging directory."""
        indexes = [
            int(match.group(1))
            for name in os.listdir(self._directory)
            if (match := self._numbered.match(name)) is not None
        ]
        index = max(indexes, default=-1) + 1

        while True:
            staging = self._directory / f'.staging-{self.name(index)}'

            try:
                staging.mkdir()
            except FileExistsError:
                index += 1
            else:
                return index, staging

    def commit(self, staging: Path) -> list[Path]:
        """Link every file in `staging` into the directory, or none of them.

        Raises:
            FileExistsError: if a name is already taken, after removing what was linked.
        """
        linked: list[Path] = []

        try:
            for source in sorted(staging.iterdir()):
                target = self._directory / source.name
                _link_exclusively(source, target)
                linked.append(target)
        except FileExistsError:
            for target in linked:
                target.unlink(missing_ok=True)

            raise

        return linked


def _link_exclusively(source: Path, target: Path) -> None:
    """Give `source` the name `target`, failing if `target` exists.

    A hard link, or an exclusive-create copy where the filesystem has no hard links.
    """
    try:
        os.link(source, target)
    except FileExistsError:
        raise
    except OSError:
        with source.open('rb') as reader, target.open('xb') as writer:
            shutil.copyfileobj(reader, writer)


def _find_synthetic_strategy(
    generators: PluginCatalog[ObjectGenerator], name: str
) -> Plugin[ObjectGenerator]:
    """The object model `name` names, if it needs no measured data.

    Raises:
        OptionsError: if no such model exists, or it reads diffraction data.
    """
    plugin = generators.find_plugin(name)

    if plugin is None or plugin.strategy.reads_diffraction_data:
        strategies = sorted(
            candidate.simple_name
            for candidate in generators
            if not candidate.strategy.reads_diffraction_data
        )
        raise OptionsError(f'"{name}" is not a synthetic strategy; choose one of {strategies}.')

    return plugin


def _format_strategy_options(plugin: Plugin[ObjectGenerator]) -> str:
    return format_options_document(
        f'{plugin.display_name} object options',
        {'synthetic': plugin.simple_name},
        plugin.strategy.get_default_options(),
    )


class _PrintOptionsAction(argparse.Action):
    """Print one strategy's default options and exit, during parsing.

    An action rather than a flag ``main`` checks, so that it works without the inputs a run
    needs, and prints before anything else the command would do.
    """

    def __init__(
        self,
        option_strings: Sequence[str],
        dest: str,
        *,
        generators: PluginCatalog[ObjectGenerator],
        **kwargs: Any,
    ) -> None:
        self._generators = generators
        super().__init__(option_strings, dest, default=argparse.SUPPRESS, **kwargs)

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence[Any] | None,
        option_string: str | None = None,
    ) -> None:
        try:
            plugin = _find_synthetic_strategy(self._generators, str(values))
        except OptionsError as exc:
            parser.error(str(exc))

        # Explicitly stdout, so ``--print-options ... > options.yaml`` captures it;
        # ``parser.exit(message=...)`` would write to stderr.
        parser._print_message(_format_strategy_options(plugin), sys.stdout)
        parser.exit()


def _build_parser(registry: PluginRegistry) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Write training data from a measured scan, or synthesized from its '
        'product with --synthetic.',
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        '-i',
        '--input-directory',
        metavar='INPUT_DIR',
        type=DirectoryType(must_exist=True),
        help='Prepared directory holding product.h5, and diffraction.h5 unless --synthetic. '
        'It is not modified.',
    )
    source.add_argument(
        '--product',
        metavar='UUID',
        type=UUID,
        help='Uuid of a product in the storage root, read with the diffraction data it '
        'derives from.',
    )
    parser.add_argument(
        '--storage-root',
        metavar='STORAGE_ROOT',
        type=Path,
        default=os.environ.get(STORAGE_ROOT_ENVIRONMENT_VARIABLE),
        help=f'Storage root holding --product. Defaults to ${STORAGE_ROOT_ENVIRONMENT_VARIABLE}.',
    )
    parser.add_argument(
        '-o',
        '--output-directory',
        metavar='OUTPUT_DIR',
        required=True,
        type=DirectoryType(must_exist=False),
        help='Training directory to add numbered files to; created if missing. Existing '
        'files are never overwritten.',
    )
    parser.add_argument(
        '--file-type',
        required=True,
        help='Training-data format: '
        f'{registry.training_data_file_writers.stringify_plugin_names()}.',
    )
    parser.add_argument(
        '--stem',
        default='train_data',
        help='File name stem; files are named <stem>_NNNN. Default: %(default)s.',
    )
    parser.add_argument(
        '--index-filter',
        choices=tuple(member.name.lower() for member in PositionIndexFilter),
        help='Scan index subset of a measured export. Default: all.',
    )
    parser.add_argument(
        '--synthetic',
        metavar='GENERATOR',
        help='Synthesize the data, generating objects with this object generator instead of '
        'exporting the measured patterns. dead_leaves gives natural-image statistics.',
    )
    parser.add_argument(
        '--count',
        type=positive_int,
        help='Number of synthetic realizations, each its own file. Default: 1.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        help='Seed for the synthetic realizations; drawn fresh and logged when omitted.',
    )
    parser.add_argument(
        '--position-jitter-m',
        type=float,
        help='Standard deviation of the Gaussian offset added to each synthetic position '
        'along each axis, in meters. Default: none.',
    )
    parser.add_argument(
        '--options',
        type=Path,
        help='YAML naming the strategy and its options, as --print-options writes; "-" reads '
        'stdin. Fields it omits keep their defaults.',
    )
    parser.add_argument(
        '--print-options',
        action=_PrintOptionsAction,
        generators=registry.object_generators,
        metavar='STRATEGY',
        help='Print the default options of a synthetic strategy as YAML and exit.',
    )
    add_log_level_argument(parser)
    parser.add_argument(
        '-v',
        '--version',
        action='version',
        version=ptychodus.VERSION_STRING,
    )
    return parser


def _resolve_strategy_options(
    plugin: Plugin[ObjectGenerator], options_file: Path | None, strategy_name: str
) -> BaseModel:
    """The strategy's options, from `options_file` when given.

    Raises:
        OptionsError: if the document names another strategy or holds unusable fields.
        OSError: if `options_file` cannot be read.
    """
    defaults = plugin.strategy.get_default_options()

    if options_file is None:
        return defaults

    selectors, fields = read_options_document(options_file, ('synthetic',))

    if selectors['synthetic'].casefold() not in (
        plugin.simple_name.casefold(),
        plugin.display_name.casefold(),
    ):
        raise OptionsError(
            f'{options_file} configures "{selectors["synthetic"]}", not "{strategy_name}".'
        )

    return apply_options_document(defaults, fields, subject=plugin.display_name)


def _jitter_positions(
    rng: numpy.random.Generator, positions: ProbePositionSequence, deviation_m: float
) -> ProbePositionSequence:
    offsets_m = rng.normal(0.0, deviation_m, size=(2, len(positions)))
    return ProbePositionSequence(
        [
            replace(position, x_m=position.x_m + dx_m, y_m=position.y_m + dy_m)
            for position, dx_m, dy_m in zip(positions, offsets_m[0], offsets_m[1])
        ]
    )


def _compute_canvas_geometry(product: Product, positions: ProbePositionSequence) -> ObjectGeometry:
    """The product's object extent on the probe's pixels, widened to cover `positions`.

    Widened about the same center rather than shifted or clipped, so that perturbed
    positions keep their whole footprint on the canvas.
    """
    probe_geometry = product.probes.get_geometry()
    product_geometry = product.object_.get_geometry()
    required = compute_object_geometry(positions, probe_geometry)
    half_width_m = max(
        0.5 * product_geometry.width_m,
        abs(required.center_x_m - product_geometry.center_x_m) + 0.5 * required.width_m,
    )
    half_height_m = max(
        0.5 * product_geometry.height_m,
        abs(required.center_y_m - product_geometry.center_y_m) + 0.5 * required.height_m,
    )
    return ObjectGeometry(
        width_px=math.ceil(2.0 * half_width_m / probe_geometry.pixel_width_m),
        height_px=math.ceil(2.0 * half_height_m / probe_geometry.pixel_height_m),
        pixel_width_m=probe_geometry.pixel_width_m,
        pixel_height_m=probe_geometry.pixel_height_m,
        center_x_m=product_geometry.center_x_m,
        center_y_m=product_geometry.center_y_m,
    )


def _synthesize(
    rng: numpy.random.Generator,
    product: Product,
    generator: ObjectGenerator,
    options: BaseModel,
    *,
    position_jitter_m: float,
) -> ReconstructInput:
    """One synthetic realization: a generated object seen through the product's probe."""
    positions = product.probe_positions

    if position_jitter_m > 0.0:
        positions = _jitter_positions(rng, positions, position_jitter_m)

    canvas = _compute_canvas_geometry(product, positions)
    object_ = generator.generate(
        options,
        _CanvasGeometryProvider(product.metadata, positions, canvas),
        rng=rng,
        probe=product.probes.get_mean_probe() if generator.reads_probe else None,
    )
    truth = replace(product, object_=object_, probe_positions=positions, losses=[])
    data = generate_diffraction_data(truth, rng)
    return ReconstructInput(
        diffraction_patterns=data.get_patterns(),
        bad_pixels=data.get_bad_pixels(),
        product=truth,
    )


def _write(
    directory: _TrainingDirectory,
    writer: TrainingDataFileWriter,
    extension: str,
    index: int,
    staging: Path,
    reconstruct_input: ReconstructInput,
) -> list[Path]:
    """Write one input under its reserved index and link it into place."""
    writer.write(staging / f'{directory.name(index)}{extension}', reconstruct_input)
    written = directory.commit(staging)
    logger.info('Wrote ' + ', '.join(path.name for path in written))
    return written


def main() -> ExitCode:
    # Before the parser, so its help names what is actually installed.
    registry = PluginRegistry.load_plugins()
    parser = _build_parser(registry)
    args = parser.parse_args()

    strategy_name: str | None = args.synthetic

    if strategy_name is None:
        misplaced = [
            flag
            for flag, value in (
                ('--count', args.count),
                ('--seed', args.seed),
                ('--position-jitter-m', args.position_jitter_m),
                ('--options', args.options),
            )
            if value is not None
        ]

        if misplaced:
            parser.error(f'{", ".join(misplaced)} applies only with --synthetic.')
    elif args.index_filter is not None:
        parser.error('--index-filter applies only to a measured export, not with --synthetic.')

    if args.position_jitter_m is not None and args.position_jitter_m < 0.0:
        parser.error('--position-jitter-m must not be negative.')

    writer_plugin = registry.training_data_file_writers.find_plugin(args.file_type)

    if writer_plugin is None:
        parser.error(
            f'No training-data format "{args.file_type}"; choose one of '
            f'{registry.training_data_file_writers.stringify_plugin_names()}.'
        )

    input_directory: Path | None = args.input_directory
    output_directory: Path = args.output_directory

    if input_directory is None:
        if args.storage_root is None:
            parser.error(f'--product needs --storage-root or ${STORAGE_ROOT_ENVIRONMENT_VARIABLE}.')

        try:
            inputs = resolve_storage_inputs(StorageLayout(Path(args.storage_root)), args.product)
        except (LayoutError, ManifestLoadError) as exc:
            parser.error(str(exc))

        diffraction_file = inputs.diffraction_file
        product_file = inputs.product_file
    else:
        diffraction_file = StandardFileLayout.DIFFRACTION.path(input_directory)
        product_file = StandardFileLayout.PRODUCT.path(input_directory)
        required_files = [product_file] if strategy_name else [diffraction_file, product_file]

        for required_file in required_files:
            if not required_file.is_file():
                parser.error(
                    f'{input_directory} holds no {required_file.name}; is it a prepared directory?'
                )

    if output_directory.resolve() == product_file.parent.resolve():
        parser.error('OUTPUT_DIR must differ from the input directory.')

    # Before any data are read, so a strategy or options that cannot be used fail in seconds.
    plugin: Plugin[ObjectGenerator] | None = None
    options: BaseModel | None = None

    if strategy_name is not None:
        try:
            plugin = _find_synthetic_strategy(registry.object_generators, strategy_name)
            options = _resolve_strategy_options(plugin, args.options, strategy_name)
        except (OptionsError, OSError) as exc:
            parser.error(str(exc))

    configure_logging(args)

    writer = writer_plugin.strategy
    extension = sorted(writer.get_file_extensions())[0]
    output_directory.mkdir(parents=True, exist_ok=True)
    directory = _TrainingDirectory(output_directory, args.stem)
    directory.warn_about_staging()

    logger.info(f'Reading {product_file}')
    product = load_product(product_file)

    try:
        if plugin is None or options is None:
            logger.info(f'Reading {diffraction_file}')
            assembled_data = load_diffraction_data(diffraction_file)
            index_filter = PositionIndexFilter[(args.index_filter or 'all').upper()]
            reconstruct_input = prepare_reconstruct_input(
                assembled_data, product, index_filter=index_filter
            )
            index, staging = directory.reserve()

            try:
                _write(directory, writer, extension, index, staging, reconstruct_input)
            finally:
                shutil.rmtree(staging, ignore_errors=True)
        else:
            if product.object_.num_layers > 1:
                logger.warning(
                    f'The product has {product.object_.num_layers} object layers; each '
                    'synthetic object has one, so the layer spacing is dropped.'
                )

            seed = numpy.random.SeedSequence(args.seed)
            logger.info(f'Synthesizing with {plugin.simple_name}, seed {seed.entropy}.')

            for _ in range(args.count or 1):
                index, staging = directory.reserve()

                try:
                    rng = numpy.random.default_rng(
                        numpy.random.SeedSequence(seed.entropy, spawn_key=(index,))
                    )
                    reconstruct_input = _synthesize(
                        rng,
                        product,
                        plugin.strategy,
                        options,
                        position_jitter_m=args.position_jitter_m or 0.0,
                    )
                    _write(directory, writer, extension, index, staging, reconstruct_input)
                finally:
                    shutil.rmtree(staging, ignore_errors=True)
    except FileExistsError as exc:
        logger.error(f'{exc.filename} already exists; nothing was overwritten.')
        return ExitCode.FAILURE

    return ExitCode.SUCCESS


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python
"""Import prepared data into a storage root, where the other commands and the webapp find it.

Each subcommand loads one kind of input, checks that it reads, and publishes it as a new
resource that carries no run -- imported data, recorded as it was given:

- ``prepared DIR``: a prepared directory, as a driver's ``--prepare-only`` writes it. Its
  ``diffraction.h5`` becomes a diffraction resource, with the scan metadata its
  ``product.h5`` carries, and the product a product resource derived from it, so
  ``ptychodus-reconstruct --product`` reconstructs it. A ``reconstruct_options.yaml`` there
  is kept with the product, and ``--product`` reconstructs with it unless given others.
- ``diffraction FILE``: a lone ``diffraction.h5``. The file holds no scan metadata, so the
  manifest takes whatever the options give.
- ``product FILE``: a ``product.h5``, such as a reconstruction made elsewhere, derived from
  a diffraction resource already in the storage root.
- ``fluorescence FILE``: a measured fluorescence dataset, in any format a fluorescence
  reader plugin reads. It is linked to nothing; ``ptychodus-enhance-xrf`` takes its uuid
  beside a product's.

Each resource's uuid is printed on its own line as it is published -- for a prepared
directory the diffraction resource's, then the product's. Nothing of an input's origin is
recorded, and the input itself is only read.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from functools import partial
from pathlib import Path
from uuid import UUID

import ptychodus
from ptychodus.api.diffraction import Polarization
from ptychodus.api.exit_codes import ExitCode
from ptychodus.api.fluorescence import FluorescenceFileReader
from ptychodus.api.io import StandardFileLayout, load_diffraction_data, load_product
from ptychodus.api.options import OptionsError, parse_options_document
from ptychodus.api.plugins import PluginCatalog, PluginRegistry
from ptychodus.api.storage import (
    STORAGE_ROOT_ENVIRONMENT_VARIABLE,
    LayoutError,
    ManifestLoadError,
    ResourceKind,
    StorageLayout,
    save_diffraction_resource,
    save_fluorescence_resource,
    save_prepared_resources,
    save_product_resource,
)
from ptychodus.cli import (
    DirectoryType,
    add_log_level_argument,
    configure_logging,
    report_resource,
)

logger = logging.getLogger('import_resources')


def _build_parser(registry: PluginRegistry) -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        '--storage-root',
        metavar='STORAGE_ROOT',
        type=DirectoryType(must_exist=True),
        default=os.environ.get(STORAGE_ROOT_ENVIRONMENT_VARIABLE),
        help='Existing storage root to import into. '
        f'Defaults to ${STORAGE_ROOT_ENVIRONMENT_VARIABLE}.',
    )
    common.add_argument(
        '--dry-run',
        action='store_true',
        help='Load and check the input without writing anything.',
    )
    add_log_level_argument(common)

    parser = argparse.ArgumentParser(
        description='Import prepared data into a storage root as resources carrying no run.',
    )
    parser.add_argument('-v', '--version', action='version', version=ptychodus.VERSION_STRING)
    subparsers = parser.add_subparsers(dest='kind', metavar='KIND', required=True)

    prepared = subparsers.add_parser(
        'prepared',
        parents=[common],
        help='a prepared directory: diffraction.h5 and product.h5',
        description='Import a prepared directory as a diffraction resource and a product '
        'derived from it, keeping any reconstruct_options.yaml with the product.',
    )
    prepared.add_argument(
        'directory',
        metavar='DIR',
        type=DirectoryType(must_exist=True),
        help='Prepared directory holding diffraction.h5 and product.h5. Not modified.',
    )
    _add_description_arguments(prepared, 'the directory name')
    _add_mmap_argument(prepared)
    prepared.set_defaults(func=_import_prepared)

    diffraction = subparsers.add_parser(
        'diffraction',
        parents=[common],
        help='a diffraction.h5 file',
        description='Import assembled diffraction data. The file holds no scan metadata, '
        'so the manifest records only what these options give.',
    )
    diffraction.add_argument(
        'file', metavar='FILE', type=Path, help='Assembled diffraction data, as diffraction.h5.'
    )
    _add_description_arguments(diffraction, 'the file name without its suffix')
    _add_mmap_argument(diffraction)
    diffraction.add_argument('--detector-distance-m', type=float, help='Detector distance.')
    diffraction.add_argument('--photon-energy-eV', type=float, help='Photon energy.')
    diffraction.add_argument('--exposure-time-s', type=float, help='Exposure time per pattern.')
    diffraction.add_argument('--tomography-angle-deg', type=float, help='Tomography angle.')
    diffraction.add_argument('--tilt-angle-deg', type=float, help='Tilt angle.')
    diffraction.add_argument(
        '--polarization',
        type=Polarization,
        choices=tuple(Polarization),
        help='Beam polarization.',
    )
    diffraction.set_defaults(func=_import_diffraction)

    product = subparsers.add_parser(
        'product',
        parents=[common],
        help='a product.h5 file, derived from a stored diffraction resource',
        description='Import a product derived from a diffraction resource in the storage '
        'root, so ptychodus-reconstruct --product reconstructs it.',
    )
    product.add_argument('file', metavar='FILE', type=Path, help='Product, as product.h5.')
    product.add_argument(
        '--diffraction',
        metavar='UUID',
        type=UUID,
        required=True,
        help='The diffraction resource the product pairs with.',
    )
    product.set_defaults(func=_import_product)

    fluorescence = subparsers.add_parser(
        'fluorescence',
        parents=[common],
        help='a measured fluorescence dataset',
        description='Import a measured fluorescence dataset, stored in the XRF-Maps layout '
        'whatever format it is read from.',
    )
    fluorescence.add_argument(
        'file', metavar='FILE', type=Path, help='Measured dataset, in the format --file-type names.'
    )
    fluorescence.add_argument(
        '--file-type',
        choices=tuple(plugin.simple_name for plugin in registry.fluorescence_file_readers),
        default='XRF-Maps',
        help='Format the dataset is read in.',
    )
    _add_description_arguments(fluorescence, 'the file name without its suffix')
    fluorescence.set_defaults(
        func=partial(_import_fluorescence, readers=registry.fluorescence_file_readers)
    )

    return parser


def _add_description_arguments(parser: argparse.ArgumentParser, default_label: str) -> None:
    parser.add_argument('--label', help=f'Label of the resource. Defaults to {default_label}.')
    parser.add_argument('--comments', default='', help='Comments on the resource.')


def _add_mmap_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        '--mmap-file',
        metavar='MMAP_FILE',
        type=Path,
        help='Stage diffraction patterns into this memmap instead of loading into RAM.',
    )


def _report(layout: StorageLayout, kind: ResourceKind, uuid: UUID) -> None:
    report_resource(logger, kind, uuid, layout.resource_folder(kind, uuid))


def _import_prepared(args: argparse.Namespace, layout: StorageLayout) -> ExitCode:
    directory: Path = args.directory
    options_file = StandardFileLayout.RECONSTRUCT_OPTIONS.path(directory)
    options_document: str | None = None

    if options_file.is_file():
        options_document = options_file.read_text(encoding='utf-8')

        try:
            # Checked only for its shape: the library it names may be installed where the
            # product is reconstructed, though not here.
            parse_options_document(
                options_document, ('library', 'algorithm'), source=str(options_file)
            )
        except OptionsError as exc:
            logger.error('%s', exc)
            return ExitCode.USAGE

    diffraction_file = StandardFileLayout.DIFFRACTION.path(directory)
    product_file = StandardFileLayout.PRODUCT.path(directory)

    try:
        data = load_diffraction_data(diffraction_file, mmap_file=args.mmap_file)
        product = load_product(product_file)
    except (OSError, KeyError, ValueError) as exc:
        logger.error('Cannot read the prepared directory %s: %s', directory, exc)
        return ExitCode.FAILURE

    if args.dry_run:
        logger.info(
            'Would import %d patterns and a product%s from %s.',
            len(data.get_indexes()),
            '' if options_document is None else ' with its options',
            directory,
        )
        return ExitCode.SUCCESS

    diffraction_uuid, product_uuid = save_prepared_resources(
        layout,
        data,
        product,
        label=directory.resolve().name if args.label is None else args.label,
        comments=args.comments,
        options_document=options_document,
    )
    _report(layout, ResourceKind.DIFFRACTION, diffraction_uuid)
    _report(layout, ResourceKind.PRODUCT, product_uuid)
    return ExitCode.SUCCESS


def _import_diffraction(args: argparse.Namespace, layout: StorageLayout) -> ExitCode:
    file: Path = args.file

    try:
        data = load_diffraction_data(file, mmap_file=args.mmap_file)
    except (OSError, KeyError, ValueError) as exc:
        logger.error('Cannot read diffraction data from %s: %s', file, exc)
        return ExitCode.FAILURE

    if args.dry_run:
        logger.info('Would import %d patterns from %s.', len(data.get_indexes()), file)
        return ExitCode.SUCCESS

    uuid = save_diffraction_resource(
        layout,
        data,
        label=file.stem if args.label is None else args.label,
        comments=args.comments,
        detector_distance_m=args.detector_distance_m,
        photon_energy_eV=args.photon_energy_eV,
        exposure_time_s=args.exposure_time_s,
        tomography_angle_deg=args.tomography_angle_deg,
        tilt_angle_deg=args.tilt_angle_deg,
        polarization=args.polarization,
    )
    _report(layout, ResourceKind.DIFFRACTION, uuid)
    return ExitCode.SUCCESS


def _import_product(args: argparse.Namespace, layout: StorageLayout) -> ExitCode:
    file: Path = args.file

    try:
        product = load_product(file)
    except (OSError, KeyError, ValueError) as exc:
        logger.error('Cannot read a product from %s: %s', file, exc)
        return ExitCode.FAILURE

    if args.dry_run:
        logger.info('Would import the product in %s, derived from %s.', file, args.diffraction)
        return ExitCode.SUCCESS

    uuid = save_product_resource(layout, product, derived_from=[args.diffraction])
    _report(layout, ResourceKind.PRODUCT, uuid)
    return ExitCode.SUCCESS


def _import_fluorescence(
    args: argparse.Namespace,
    layout: StorageLayout,
    *,
    readers: PluginCatalog[FluorescenceFileReader],
) -> ExitCode:
    file: Path = args.file
    reader = readers.get_strategy_by_name(args.file_type)

    try:
        dataset = reader.read(file)
    except (OSError, KeyError, ValueError) as exc:
        logger.error('Cannot read a %s fluorescence dataset from %s: %s', args.file_type, file, exc)
        return ExitCode.FAILURE

    if args.dry_run:
        logger.info('Would import %d element maps from %s.', len(dataset), file)
        return ExitCode.SUCCESS

    uuid = save_fluorescence_resource(
        layout,
        dataset,
        label=file.stem if args.label is None else args.label,
        comments=args.comments,
    )
    _report(layout, ResourceKind.FLUORESCENCE, uuid)
    return ExitCode.SUCCESS


def main() -> ExitCode:
    registry = PluginRegistry.load_plugins()
    parser = _build_parser(registry)
    args = parser.parse_args()

    if args.storage_root is None:
        parser.error(
            f'Name a storage root with --storage-root or ${STORAGE_ROOT_ENVIRONMENT_VARIABLE}.'
        )

    layout = StorageLayout(args.storage_root)

    if args.kind == 'prepared':
        for required_file in (
            StandardFileLayout.DIFFRACTION.path(args.directory),
            StandardFileLayout.PRODUCT.path(args.directory),
        ):
            if not required_file.is_file():
                parser.error(
                    f'{args.directory} holds no {required_file.name}; is it a prepared directory?'
                )
    elif args.kind == 'product':
        try:
            kind = layout.locate(args.diffraction).kind
        except (LayoutError, ManifestLoadError) as exc:
            parser.error(str(exc))

        if kind != ResourceKind.DIFFRACTION:
            parser.error(f'Resource {args.diffraction} is a {kind}, not a diffraction resource.')

    configure_logging(args)
    code: ExitCode = args.func(args, layout)
    return code


if __name__ == '__main__':
    sys.exit(main())

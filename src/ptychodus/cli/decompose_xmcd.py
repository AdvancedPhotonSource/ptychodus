#!/usr/bin/env python
"""Decompose a left/right circularly polarized product pair into structural and magnetic parts.

Takes two ptychodus product files reconstructed from the same scan region with opposite
helicities and writes two product files: the helicity-independent (structural) object and
the magnetic factor, decomposed under ``O_+ = O_struct * M`` and ``O_- = O_struct / M``.
The LCP object is aligned to the RCP object and its scale, phase offset and phase ramp are
standardized against it, so both outputs are on the RCP product's frame and carry its
probe, probe positions and metadata. It reads no settings file and builds no application
model.

Nothing is written unless the decomposition completes. The magnetic contrast is relative
to the field of view: the standardization absorbs a uniform magnetic phase and amplitude.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import inspect
import logging
import signal
import sys

import ptychodus
from ptychodus.api.diffraction import Polarization
from ptychodus.api.exit_codes import ExitCode
from ptychodus.api.io import load_product, save_product
from ptychodus.api.object import RegistrationQuantity
from ptychodus.api.product import Product
from ptychodus.api.xmcd import estimate_xmcd
from ptychodus.cli import add_log_level_argument, configure_logging, positive_int

logger = logging.getLogger('decompose_xmcd')


def _non_negative_float(text: str) -> float:
    value = float(text)

    if not value >= 0.0:
        raise argparse.ArgumentTypeError(f'"{text}" must be at least 0!')

    return value


def _positive_float(text: str) -> float:
    value = float(text)

    if not value > 0.0:
        raise argparse.ArgumentTypeError(f'"{text}" must be greater than 0!')

    return value


def _registration_quantity(text: str) -> RegistrationQuantity:
    try:
        return RegistrationQuantity[text.upper()]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(f'"{text}" is not a registration quantity!') from exc


def _build_parser() -> argparse.ArgumentParser:
    # Read from the decomposition itself, so a default is stated in one place.
    defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(estimate_xmcd).parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    }
    default_quantity: RegistrationQuantity = defaults['registration_quantity']

    parser = argparse.ArgumentParser(
        description='Decompose an LCP/RCP product pair into structural and magnetic products.',
    )
    parser.add_argument(
        '--lcp',
        metavar='LCP_FILE',
        type=Path,
        required=True,
        help='Left-circularly-polarized product HDF5 file (O_-).',
    )
    parser.add_argument(
        '--rcp',
        metavar='RCP_FILE',
        type=Path,
        required=True,
        help='Right-circularly-polarized product HDF5 file (O_+), the reference frame of '
        'both outputs.',
    )
    parser.add_argument(
        '--structural',
        metavar='OUTPUT_FILE',
        type=Path,
        required=True,
        help='Structural product HDF5 file to write. An existing file is overwritten.',
    )
    parser.add_argument(
        '--magnetic',
        metavar='OUTPUT_FILE',
        type=Path,
        required=True,
        help='Magnetic product HDF5 file to write. An existing file is overwritten.',
    )
    parser.add_argument(
        '--upsample-factor',
        type=positive_int,
        default=defaults['upsample_factor'],
        help='Sub-pixel precision of the alignment, as 1/N pixel. '
        f'Defaults to {defaults["upsample_factor"]}.',
    )
    parser.add_argument(
        '--registration-quantity',
        type=_registration_quantity,
        metavar='{' + ','.join(quantity.name.lower() for quantity in RegistrationQuantity) + '}',
        default=default_quantity,
        help='Quantity correlated to align the objects. '
        f'Defaults to {default_quantity.name.lower()}.',
    )
    parser.add_argument(
        '--high-pass-sigma-px',
        type=_non_negative_float,
        default=defaults['high_pass_sigma_px'],
        help='Gaussian background sigma removed before alignment, 0 to disable. '
        f'Defaults to {defaults["high_pass_sigma_px"]:g}.',
    )
    parser.add_argument(
        '--epsilon',
        type=_positive_float,
        default=defaults['epsilon'],
        help='Added to the LCP object before dividing, to keep the magnetic amplitude '
        f'finite where it is zero. Defaults to {defaults["epsilon"]:g}.',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Read and check the inputs, and report them, without decomposing.',
    )
    add_log_level_argument(parser)
    parser.add_argument(
        '-v',
        '--version',
        action='version',
        version=ptychodus.VERSION_STRING,
    )
    return parser


def _warn_if_swapped(product: Product, expected: Polarization, product_file: Path) -> None:
    polarization = product.metadata.polarization

    if polarization is not None and polarization is not expected:
        logger.warning(
            '%s records %s polarization but was passed as %s; were the inputs swapped? '
            'Swapping them inverts the magnetic contrast.',
            product_file,
            polarization.value,
            expected.value,
        )


def main() -> ExitCode:
    parser = _build_parser()
    args = parser.parse_args()

    lcp_file: Path = args.lcp
    rcp_file: Path = args.rcp
    structural_file: Path = args.structural
    magnetic_file: Path = args.magnetic
    input_files = (lcp_file, rcp_file)

    for input_file in input_files:
        if not input_file.is_file():
            parser.error(f'{input_file} is not a file.')

    if structural_file.resolve() == magnetic_file.resolve():
        parser.error('--structural and --magnetic must name different files.')

    for output_file in (structural_file, magnetic_file):
        if output_file.exists():
            if output_file.is_dir():
                parser.error(f'{output_file} is a directory; name the file to write.')

            for input_file in input_files:
                if output_file.samefile(input_file):
                    parser.error(f'{output_file} must differ from the input {input_file}.')

    configure_logging(args)

    logger.info('Reading LCP %s', lcp_file)
    lcp_product = load_product(lcp_file)
    logger.info('Reading RCP %s', rcp_file)
    rcp_product = load_product(rcp_file)

    _warn_if_swapped(lcp_product, Polarization.LEFT_CIRCULAR, lcp_file)
    _warn_if_swapped(rcp_product, Polarization.RIGHT_CIRCULAR, rcp_file)
    logger.info(
        'Upsample factor %d, registration quantity %s, high-pass sigma %g px, epsilon %g',
        args.upsample_factor,
        args.registration_quantity.name.lower(),
        args.high_pass_sigma_px,
        args.epsilon,
    )

    if args.dry_run:
        for label, product in (('LCP', lcp_product), ('RCP', rcp_product)):
            logger.info(
                'Dry run: %s object %s, pixel %s, polarization %s.',
                label,
                product.object_.get_array().shape,
                product.object_.get_pixel_geometry(),
                product.metadata.polarization,
            )

        logger.info('Would write %s and %s', structural_file, magnetic_file)
        return ExitCode.SUCCESS

    # A scheduler ends a job with SIGTERM. Raised as an interrupt it unwinds through the
    # cancellation path below, so nothing is written.
    signal.signal(signal.SIGTERM, signal.default_int_handler)

    try:
        result = estimate_xmcd(
            lcp_product,
            rcp_product,
            upsample_factor=args.upsample_factor,
            registration_quantity=args.registration_quantity,
            high_pass_sigma_px=args.high_pass_sigma_px,
            epsilon=args.epsilon,
        )
    except KeyboardInterrupt:
        logger.warning('XMCD cancelled; nothing written.')
        return ExitCode.CANCELLED
    except Exception:
        logger.exception('XMCD failed; nothing written.')
        return ExitCode.FAILURE

    for output_file, product in (
        (structural_file, result.structural_product),
        (magnetic_file, result.magnetic_product),
    ):
        if output_file.exists():
            logger.warning('%s will be overwritten.', output_file)

        output_file.parent.mkdir(parents=True, exist_ok=True)
        save_product(output_file, product)
        logger.info('Wrote %s', output_file)

    return ExitCode.SUCCESS


if __name__ == '__main__':
    sys.exit(main())

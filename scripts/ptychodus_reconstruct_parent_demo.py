#!/usr/bin/env python
"""Reference launcher for ``scripts/ptychodus_reconstruct.py``.

Shows the whole contract in one place: build a pty-chi options object from
ptychodus settings exactly as the GUI would, derive the ``--reconstructor``
token *from that object*, hand the serialized options to the child on stdin,
and consume its JSON event stream. Optionally cancels the run part-way to
exercise the signal path.

    python scripts/ptychodus_reconstruct_parent_demo.py \\
        --diffraction-input staging/diffraction.h5 \\
        --product-input     staging/product-in.h5 \\
        --settings          staging/settings.ini \\
        --product-output    out/product-out.h5 \\
        --num-epochs 6 [--cancel-after-s 10]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

from ptychodus.api.io import load_diffraction_data, load_product
from ptychodus.api.reconstruct import prepare_reconstruct_input
from ptychodus.api.settings import SettingsRegistry
from ptychodus.model.processing.api import ProcessingAlgorithmParameter
from ptychodus.model.processing.settings import ProcessingSettings
from ptychodus.model.ptychi.core import PtyChiReconstructorLibrary
from ptychodus.model.ptychi.helper import PtyChiOptionsHelper
from ptychodus.model.ptychi.reconstructor import PtyChiSettingsBundle, create_task_options
from ptychodus.model.ptychi.task import dump_task_options, reconstructor_argument

CHILD = Path(__file__).parent / 'ptychodus_reconstruct.py'


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--diffraction-input',
        metavar='DIFFRACTION_INPUT_FILE',
        required=True,
        type=Path,
    )
    parser.add_argument(
        '--product-input',
        metavar='PRODUCT_INPUT_FILE',
        required=True,
        type=Path,
    )
    parser.add_argument(
        '-s',
        '--settings',
        metavar='SETTINGS_FILE',
        required=True,
        type=Path,
    )
    parser.add_argument(
        '--product-output',
        metavar='PRODUCT_OUTPUT_FILE',
        required=True,
        type=Path,
    )
    parser.add_argument('--num-epochs', type=int)
    parser.add_argument('--num-sync-epochs', default=1, type=int)
    parser.add_argument(
        '--cancel-after-s',
        metavar='TIME',
        type=float,
        help='Seconds after the child reports "started" before sending SIGTERM.',
    )
    return parser.parse_args()


def _build_task_options(args: argparse.Namespace):
    """Build the options the way ptychodus itself does, from a settings INI."""
    registry = SettingsRegistry()
    registry.open_settings(args.settings)
    library = PtyChiReconstructorLibrary(registry, is_developer_mode_enabled=False)

    options_helper = PtyChiOptionsHelper(
        library.settings,
        library.object_settings,
        library.probe_settings,
        library.probe_position_settings,
        library.opr_settings,
    )
    bundle = PtyChiSettingsBundle(
        dm=library.dm_settings,
        raar=library.raar_settings,
        pie=library.pie_settings,
        lsqml=library.lsqml_settings,
        autodiff=library.autodiff_settings,
        bh=library.bh_settings,
    )

    # The chosen algorithm lives in the settings as '<library>_<reconstructor>'
    # (e.g. 'pty-chi_lsqml'); split off the library half and hand the
    # reconstructor name to create_task_options, which case-folds it.
    processing_settings = ProcessingSettings(registry)
    _library, algorithm = ProcessingAlgorithmParameter.split_key(
        processing_settings.algorithm.get_value()
    )

    # create_task_options still takes a full ReconstructInput, though after the
    # pty-chi v2.0 move the builders only read the product. Loading the
    # diffraction data here is therefore pure overhead for a launcher -- worth
    # narrowing that signature to Product.
    reconstruct_input = prepare_reconstruct_input(
        load_diffraction_data(args.diffraction_input),
        load_product(args.product_input),
    )
    return create_task_options(algorithm, options_helper, bundle, reconstruct_input)


def main() -> int:
    args = _parse_arguments()

    task_options = _build_task_options(args)

    if args.num_epochs is not None:
        task_options.reconstructor_options.num_epochs = args.num_epochs

    # The token and the blob must come from the same object. Never a literal:
    # naming a class with a different field set fails loudly child-side, but
    # PIE, ePIE and rPIE are indistinguishable once serialized.
    reconstructor = reconstructor_argument(task_options)

    command = [
        sys.executable,
        str(CHILD),
        '--reconstructor',
        reconstructor,
        '--diffraction-input',
        str(args.diffraction_input),
        '--product-input',
        str(args.product_input),
        '--product-output',
        str(args.product_output),
        '--num-sync-epochs',
        str(args.num_sync_epochs),
    ]

    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )

    assert process.stdin is not None
    assert process.stdout is not None

    process.stdin.write(dump_task_options(task_options))
    process.stdin.close()

    started_s = time.monotonic()

    for line in process.stdout:
        line = line.strip()

        if not line:
            continue

        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            print(f'[{time.monotonic() - started_s:7.2f}s] NON-JSON ON STDOUT: {line!r}')
            continue

        name = event.pop('event')
        print(f'[{time.monotonic() - started_s:7.2f}s] {name:<11} {event}')

        # Arm the cancel timer only once the run is live. A signal sent while
        # the child is still importing torch cannot be caught and kills it
        # outright, with no partial result and no final event.
        if name == 'started' and args.cancel_after_s is not None:
            timer = threading.Timer(args.cancel_after_s, process.terminate)
            timer.daemon = True
            timer.start()

    returncode = process.wait()
    print(f'child exited with {returncode}')
    return returncode


if __name__ == '__main__':
    sys.exit(main())

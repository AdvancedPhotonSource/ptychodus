#!/usr/bin/env python
"""Reference launcher for ``scripts/ptychodus_reconstruct.py``.

Shows the whole contract in one place: build a pty-chi options object from
ptychodus settings exactly as the GUI would, serialize it, and hand the blob to
the child on stdin. The envelope in :func:`dump_task_options` carries the
reconstructor identity, so nothing algorithm-specific rides on argv.
Optionally cancels the run part-way to exercise the signal path.

    python scripts/ptychodus_reconstruct_parent_demo.py \\
        --diffraction-input staging/diffraction.h5 \\
        --product-input     staging/product.h5 \\
        --settings          staging/settings.ini \\
        --product-output    out/product.h5 \\
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

from ptychodus.api.io import load_product
from ptychodus.api.settings import SettingsRegistry
from ptychodus.model.processing.api import ProcessingAlgorithmParameter
from ptychodus.model.processing.settings import ProcessingSettings
from ptychodus.model.ptychi.core import PtyChiReconstructorLibrary
from ptychodus.model.ptychi.task import dump_task_options

CHILD = Path(__file__).parent / 'ptychodus_reconstruct.py'


def main() -> int:
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
    args = parser.parse_args()

    # Build the options the way ptychodus itself does, from a settings INI.
    registry = SettingsRegistry()
    registry.open_settings(args.settings)
    library = PtyChiReconstructorLibrary(registry, is_developer_mode_enabled=False)

    # The chosen algorithm lives in the settings as '<library>_<reconstructor>'
    # (e.g. 'pty-chi_lsqml'); split off the library half and hand the
    # reconstructor name to library.build_task_options, which case-folds it.
    processing_settings = ProcessingSettings(registry)
    _library, algorithm = ProcessingAlgorithmParameter.split_key(
        processing_settings.algorithm.get_value()
    )

    task_options = library.build_task_options(algorithm, load_product(args.product_input))

    if args.num_epochs is not None:
        task_options.reconstructor_options.num_epochs = args.num_epochs

    command = [
        sys.executable,
        str(CHILD),
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

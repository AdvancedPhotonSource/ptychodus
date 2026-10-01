#!/usr/bin/env python
"""Reference launcher for the ``ptychodus-reconstruct-subprocess`` command.

Shows the whole contract in one place: build a pty-chi options object from
ptychodus settings exactly as the GUI would, serialize it, and hand the blob to
the child on stdin. The ``options_class_name`` that pty-chi stamps into the
serialized dict carries the algorithm identity, so nothing algorithm-specific
rides on argv.
Optionally cancels the run part-way to exercise the signal path.

    python scripts/ptychodus_reconstruct_parent_demo.py \\
        -i staging/ -o out/ \\
        --settings staging/settings.ini \\
        --num-epochs 6 [--cancel-after-s 10]

The input directory is the ptychodus standard layout, so it is whatever an
earlier run or staging step wrote: ``diffraction.h5`` and ``product.h5``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from ptychodus.api.io import StandardFileLayout, load_product
from ptychodus.api.settings import SettingsRegistry
from ptychodus.model.processing.api import ProcessingAlgorithmParameter
from ptychodus.model.processing.settings import ProcessingSettings
from ptychodus.model.ptychi.core import PtyChiReconstructorLibrary
from ptychodus.model.ptychi.task import dump_task_options

COMMAND_NAME = 'ptychodus-reconstruct-subprocess'


def _child_command() -> list[str]:
    """How to start the child, installed or not.

    The console script is the supported entry point, but a checkout that has not been
    installed has no such executable on PATH, and running the module directly is the
    same code by another name.
    """
    executable = shutil.which(COMMAND_NAME)

    if executable is not None:
        return [executable]

    return [sys.executable, '-m', 'ptychodus.cli.reconstruct_subprocess']


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '-i',
        '--input-directory',
        metavar='INPUT_DIR',
        required=True,
        type=Path,
    )
    parser.add_argument(
        '-o',
        '--output-directory',
        metavar='OUTPUT_DIR',
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

    product_file = StandardFileLayout.PRODUCT.path(args.input_directory)
    task_options = library.build_task_options(algorithm, load_product(product_file))

    if args.num_epochs is not None:
        task_options.reconstructor_options.num_epochs = args.num_epochs

    command = [
        *_child_command(),
        '-i',
        str(args.input_directory),
        '-o',
        str(args.output_directory),
        # The options were built in memory here, so there is no file to point at. Saying
        # so explicitly keeps the handoff visible rather than implied by an empty flag.
        '--ptychi-options-file',
        '-',
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

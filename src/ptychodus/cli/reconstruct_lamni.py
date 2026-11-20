#!/usr/bin/env python
"""Reconstruct one APS 31-ID-E LamNI ptychography dataset through the ptychodus api.

LamNI records much of its geometry in the HDF5 itself, so the fallbacks below are often
unused. Position files come in three header layouts (Orchestra, softGlueZynq raw and
processed), all handled by one reader that carries a real per-frame index rather than relying
on array order. The beam center is taken from the command line, then the file, and is
estimated from the patterns only when neither gives one.

Scope and limitations
---------------------

- One scan, stock LSQML unless ``--options`` names something else.
- Output is a ptychodus product HDF5. Feeding it back as ``--probe-file`` on the next
  scan is the intended warm start.
- No GPU selection or thread-count side effects. Choose a device with
  ``CUDA_VISIBLE_DEVICES`` in the environment.
- Positions carry their file-provided index, so patterns and positions pair through
  :func:`prepare_reconstruct_input` rather than by array order.
- Geometry resolves in one order -- command line, then the file, then a built-in
  fallback -- and the source of each value is logged.
"""

from __future__ import annotations

import sys

from ptychodus.api.exit_codes import ExitCode
from ptychodus.plugins.aps31id_lamni.diffraction_file import LamNIDiffractionFileReader
from ptychodus.cli._reconstruct_standard import (
    InstrumentProfile,
    run_standard_reconstruction,
)

PROFILE = InstrumentProfile(
    logger_name='reconstruct_lamni',
    description='APS 31-ID-E LamNI ptychography reconstruction via the ptychodus api.',
    product_name='lamni-reconstruct',
    diffraction_reader='APS_LamNI',
    position_reader='APS_LamNI',
    diffraction_file_help='Raw LamNI HDF5 file (APS 31-ID-E).',
    position_file_help='LamNI probe-position .dat file (Orchestra or softGlueZynq).',
    fzp_preset_help=(
        'Zone-plate preset to generate the initial probe from, e.g. "APS 31-ID-E LamNI". Empty '
        'by default, so the initial probe is generated from the mean pattern. Ignored with '
        '--probe-file.'
    ),
    default_invalid_counts=LamNIDiffractionFileReader.INVALID_COUNTS,
)


def main() -> ExitCode:
    return run_standard_reconstruction(PROFILE)


if __name__ == '__main__':
    sys.exit(main())

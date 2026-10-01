#!/usr/bin/env python
"""Reconstruct one APS 2-ID-E XFM ptychography dataset through the ptychodus api.

The XFM instrument at 2-ID-E writes one Eiger HDF5 per scan line beside an EPICS MDA file
holding the positions. Both the scan number and the line number are three digits wide, so the
series is identified by the trailing field.

The positioner is driven over more points per line than the detector records, so both readers
number their output line-major -- `line * stride + column` -- rather than by running count.
Pairing then falls out of the scan index: the surplus commanded points go unclaimed, a line the
detector cut short claims fewer, and a detector line past the end of the positioner record has
no position and is dropped.

Scope and limitations
---------------------

- One scan, stock LSQML unless ``--ptychi-options-file`` names something else.
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
from ptychodus.cli._reconstruct_standard import (
    InstrumentProfile,
    run_standard_reconstruction,
)

PROFILE = InstrumentProfile(
    logger_name='reconstruct_2ide',
    description='APS 2-ID-E XFM ptychography reconstruction via the ptychodus api.',
    product_name='2ide-reconstruct',
    diffraction_reader='APS_2IDE',
    position_reader='APS_2IDE',
    diffraction_file_help='Any member of the fly###_data_NNN.h5 series; the rest are globbed.',
    position_file_help='EPICS MDA file, normally mda/2xfm_NNNN.mda.',
    fzp_preset_help=(
        'FresnelZonePlate plugin preset for the model probe. No zone-plate preset is '
        'verified for this instrument, so this is empty by default and the probe is '
        'estimated from the data instead. Ignored with --probe-file.'
    ),
    default_detector_distance_m=2.12,
)


def main() -> ExitCode:
    return run_standard_reconstruction(PROFILE)


if __name__ == '__main__':
    sys.exit(main())

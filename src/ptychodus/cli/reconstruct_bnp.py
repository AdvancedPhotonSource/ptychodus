#!/usr/bin/env python
"""Reconstruct one APS 2-ID-D Bionanoprobe ptychography dataset through the ptychodus api.

The Bionanoprobe writes one Eiger HDF5 per scan line beside an EPICS MDA file. Its positions
are in micrometers, where the 2-ID-E microprobe uses millimeters, and its frames carry enough
invalid-pixel markers that the beam center cannot be estimated without cutting them first.

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
    logger_name='reconstruct_bnp',
    description='APS 2-ID-D Bionanoprobe ptychography reconstruction via the ptychodus api.',
    product_name='bnp-reconstruct',
    diffraction_reader='APS_BNP',
    position_reader='APS_BNP',
    diffraction_file_help='Any member of the bnp_flyNNNN_NNNNNN.h5 series; the rest are globbed.',
    position_file_help='EPICS MDA file, normally mda/bnp_flyNNNN.mda.',
    fzp_preset_help=(
        'FresnelZonePlate plugin preset for the model probe. No zone-plate preset is '
        'verified for this instrument, so this is empty by default and the probe is '
        'estimated from the data instead. Ignored with --probe-file.'
    ),
    default_detector_distance_m=2.06,
)


def main() -> ExitCode:
    return run_standard_reconstruction(PROFILE)


if __name__ == '__main__':
    sys.exit(main())

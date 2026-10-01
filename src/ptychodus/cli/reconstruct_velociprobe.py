#!/usr/bin/env python
"""Reconstruct one APS 33-ID-C VelociProbe ptychography dataset through the ptychodus api.

The VelociProbe master file records its own detector distance, energy, pixel pitch and beam
center, so the built-in fallbacks below are rarely reached. It is the one instrument here with
a verified zone-plate preset.

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
    logger_name='reconstruct_velociprobe',
    description='APS 33-ID-C VelociProbe ptychography reconstruction via the ptychodus api.',
    product_name='velociprobe-reconstruct',
    diffraction_reader='APS_Velociprobe',
    position_reader='APS_Velociprobe_PE',
    diffraction_file_help='VelociProbe master HDF5 file (fly###_master.h5).',
    position_file_help='Position-encoder text file (fly###_0.txt).',
    fzp_preset_help=(
        'FresnelZonePlate plugin preset for the model probe. Defaults to the VelociProbe '
        'zone plate. Ignored with --probe-file.'
    ),
    default_fzp_preset='APS_Velociprobe',
)


def main() -> ExitCode:
    return run_standard_reconstruction(PROFILE)


if __name__ == '__main__':
    sys.exit(main())

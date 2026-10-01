#!/usr/bin/env python
"""Reconstruct one APS 19-ID-E In-situ Nanoprobe ptychography dataset through the ptychodus api.

The In-situ Nanoprobe focuses with KB mirrors. Its fly scans expose a single trajectory
positioner; its per-trigger X/Y readback is averaged downstream into a companion
``Processed/SOCKETSERVER/Scan_NNNN_position.h5`` file alongside the
``Raw/Scan_NNNN/PTYCHO/`` diffraction series, read by the ``APS_ISN`` position reader.

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
    logger_name='reconstruct_isn',
    description=(
        'APS 19-ID-E In-situ Nanoprobe ptychography reconstruction via the ptychodus api.'
    ),
    product_name='isn-reconstruct',
    diffraction_reader='APS_ISN',
    position_reader='APS_ISN',
    diffraction_file_help=(
        'Any member of the PTYCHO/scan_NNNN_FFFFF.h5 (or older 19ide_NNNN_NNN.h5) series; '
        'the rest are globbed.'
    ),
    position_file_help='HDF5 file, normally Processed/SOCKETSERVER/Scan_NNNN_position.h5.',
    fzp_preset_help=(
        'FresnelZonePlate plugin preset for the model probe. This instrument focuses with '
        'KB mirrors rather than a zone plate, so a zone-plate probe would be the wrong '
        'model entirely and this is empty by default. Ignored with --probe-file.'
    ),
    default_detector_distance_m=6.16,
)


def main() -> ExitCode:
    return run_standard_reconstruction(PROFILE)


if __name__ == '__main__':
    sys.exit(main())

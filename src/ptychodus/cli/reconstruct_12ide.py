#!/usr/bin/env python
"""Reconstruct one APS 12-ID-E Ptycho-SAXS ptychography dataset through the ptychodus api.

Reads the layout written from 2026_2 on: a scan directory holding a NeXus
``<sample>_<scan>_master.h5`` beside its ``<sample>_<scan>_<NNNNN>.h5`` frame files. The master
supplies the energy, exposure, detector distance and pitch, and the nominal scan positions. The
long flight path (up to 10 m) and the Pilatus 172 um pitch make this the coarsest-sampled
instrument here. The beam center is not recorded, so pass it on the command line.

Scope and limitations
---------------------

- One scan, stock LSQML unless ``--ptychi-options`` names something else.
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
    logger_name='reconstruct_12ide',
    description='APS 12-ID-E Ptycho-SAXS ptychography reconstruction via the ptychodus api.',
    product_name='12ide-reconstruct',
    diffraction_reader='APS_PtychoSAXS',
    position_reader='APS_PtychoSAXS',
    diffraction_file_help=(
        'The scan master <sample>_<scan>_master.h5, or any of its frame files; '
        'the rest are globbed.'
    ),
    position_file_help=(
        'The scan master, or any frame file of the scan; positions are read from the master.'
    ),
    fzp_preset_help=(
        'FresnelZonePlate plugin preset for the model probe. No zone-plate preset is '
        'verified for this instrument, so this is empty by default and the probe is '
        'estimated from the data instead. Ignored with --probe-file.'
    ),
    default_detector_distance_m=10.2,
)


def main() -> ExitCode:
    return run_standard_reconstruction(PROFILE)


if __name__ == '__main__':
    sys.exit(main())

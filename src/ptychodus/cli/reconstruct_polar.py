#!/usr/bin/env python
"""Reconstruct one APS 4-ID-B,G,H POLAR ptychography dataset through the ptychodus api.

The POLAR master advertises both halves of a scan as external links under
``/entry/externals``: the Eiger frames at ``eiger/scan_NNNNNN.h5`` and, for fly scans, the
softGlueZynq position stream at ``pos_stream/scan_NNNNNN.h5``. Both readers therefore take
the *same* master path, and ``--position-file`` is normally a repeat of
``--diffraction-file``.

How much geometry the file carries depends on when it was written. The newest layout
records the detector distance, the beam center and the mono energy; older ones record only
the energy, so the distance falls back to the built-in default below and the beam center
must be given or estimated. No layout records the detector pixel pitch, which the reader
supplies as the Eiger's 75 um.

Fly scans index positions by the raw softGlueZynq trigger counter, which starts at 0, while
the diffraction reader emits 1-based frame indexes -- so frame ``k`` pairs with trigger
``k + 1`` and trigger 0 is left over. Step scans index both sides off the same Eiger unique
id and pair 1:1. Neither offset is applied by hand: ``prepare_reconstruct_input`` pairs on
the index, which is what makes both conventions work through one script.

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
- The beamline's ``4idd_data_preprocessing_flyscan_v2.py`` normalizes patterns by I0
  (``dp / i0 * 1e5``); this script does not. Ptychodus carries the per-position I0 as
  ``probe_photon_count``, which feeds the illumination map rather than the patterns.
"""

from __future__ import annotations

import sys

from ptychodus.api.exit_codes import ExitCode
from ptychodus.cli._reconstruct_standard import (
    InstrumentProfile,
    run_standard_reconstruction,
)

PROFILE = InstrumentProfile(
    logger_name='reconstruct_polar',
    description='APS 4-ID-B,G,H POLAR ptychography reconstruction via the ptychodus api.',
    product_name='polar-reconstruct',
    diffraction_reader='APS_Polar',
    position_reader='APS_Polar',
    diffraction_file_help='POLAR master HDF5 file (scan_NNNNNN_master.hdf).',
    position_file_help='POLAR master HDF5 file; normally the same path as --diffraction-file.',
    fzp_preset_help=(
        'FresnelZonePlate plugin preset for the model probe. No zone-plate preset is '
        'verified for this instrument, so this is empty by default and the probe is '
        'estimated from the data instead. Ignored with --probe-file.'
    ),
    default_detector_distance_m=1.91,
    offers_pattern_filters=True,
)


def main() -> ExitCode:
    return run_standard_reconstruction(PROFILE)


if __name__ == '__main__':
    sys.exit(main())

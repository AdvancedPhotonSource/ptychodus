#!/usr/bin/env python
"""Reconstruct one fold_slice preprocessed ptychography dataset through the ptychodus api.

Reads the cropped-and-centered pair the fold_slice preprocessing step writes, which is how
most batch reconstructions at these beamlines are actually fed. The files record no geometry
at all, so --detector-distance-m, --probe-energy-eV and --detector-pixel-size-m are
effectively required.

The patterns arrive already prepared, so this driver conditions them no further: it applies
no crop, no beam-center search, no value filter, no bad-pixel override and no flip or
transpose. Whatever fold_slice wrote is what gets reconstructed. Any of those belongs in the
preprocessing step that produced the file.

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
- The detector bad-pixel mask the reader itself supplies is still honored; only the
  ``--bad-pixels-file`` override is absent.
"""

from __future__ import annotations

import sys

from ptychodus.api.exit_codes import ExitCode
from ptychodus.cli._reconstruct_standard import (
    InstrumentProfile,
    run_standard_reconstruction,
)

PROFILE = InstrumentProfile(
    logger_name='reconstruct_foldslice',
    description='fold_slice preprocessed ptychography reconstruction via the ptychodus api.',
    product_name='foldslice-reconstruct',
    diffraction_reader='fold_slice',
    position_reader='fold_slice',
    diffraction_file_help='Preprocessed pattern file, normally data_roi<N>_Ndp<N>..._dp.hdf5.',
    position_file_help='Matching parameter file, normally the same stem with _para.hdf5.',
    fzp_preset_help=(
        'FresnelZonePlate plugin preset for the model probe. This layout is written by '
        'several instruments, so no one zone plate applies and this is empty by default. '
        'Ignored with --probe-file.'
    ),
    conditions_patterns=False,
    offers_detector_pixel_size=True,
    offers_probe_photon_count=True,
)


def main() -> ExitCode:
    return run_standard_reconstruction(PROFILE)


if __name__ == '__main__':
    sys.exit(main())

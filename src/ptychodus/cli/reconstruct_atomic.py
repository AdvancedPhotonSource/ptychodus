#!/usr/bin/env python
"""Reconstruct one APS Atomic ptychography dataset through the ptychodus api.

Atomic writes one uncompressed EPICS-areaDetector TIFF per scan point into a per-scan
directory, beside an EPICS MDA file holding the positions and a JSON list of the
detector's bad pixels. The TIFF frame counter is not zero padded, so a scan longer than
999 points rolls from ``_999`` to ``_1000`` partway through and the series is globbed at
any counter width.

Scope and limitations
---------------------

- Positions are piezo driver command voltages, not measured ones. Neither scanned axis
  records an encoder, so the APS_Atomic position reader scales volts by the driver's
  fixed 10 um/V calibration. Stage drift, creep, hysteresis and nonlinearity are all
  invisible to that conversion; position correction during reconstruction is the only
  thing that can absorb them.
- No file records the detector distance, and the TIFF wavelength tag is an
  areaDetector default rather than the monochromator. Both geometry values therefore
  come from the command line or from the built-in fallbacks, and nothing in the data
  will contradict a wrong one. The MDA does carry the mono readback as a detector
  channel, but positions are all this driver reads from it.
- The series is scoped by the filename prefix, not by the directory. A directory
  holding two scans reconstructs whichever prefix was named, and a directory whose name
  disagrees with the prefix inside it is reconstructed by the prefix.
- Pattern count and position count are not cross-checked beyond the warning assembly
  emits. A detector armed before the scan record started leaves extra leading frames,
  which shifts every pattern against its position without failing.
- A rank-3 MDA -- a raster repeated over an outer axis -- is rejected by the position
  reader rather than flattened, because its outer axis is a different stage in
  different units.
- Diffraction from this instrument is compact: most of the signal sits within a few tens
  of pixels of the beam center. Crop with --crop-extent-px rather than reconstructing
  the full frame, and expect the usable radius to bound the resolution.
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
    logger_name='reconstruct_atomic',
    description='APS Atomic ptychography reconstruction via the ptychodus api.',
    product_name='atomic-reconstruct',
    diffraction_reader='APS_Atomic',
    position_reader='APS_Atomic',
    diffraction_file_help='Any member of one scan TIFF series; the rest of the prefix is globbed.',
    position_file_help='EPICS MDA file, normally mda_files/34ide_NNNNNN.mda.',
    fzp_preset_help=(
        'FresnelZonePlate plugin preset for the model probe. No zone-plate preset is '
        'verified for this instrument, so this is empty by default and the probe is '
        'estimated from the data instead. Ignored with --probe-file.'
    ),
    default_detector_distance_m=1.04,
    default_detector_pixel_size_m=7.5e-05,
    default_bad_pixels_reader='APS_Atomic_Bad_Pixels',
)


def main() -> ExitCode:
    return run_standard_reconstruction(PROFILE)


if __name__ == '__main__':
    sys.exit(main())

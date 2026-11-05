#!/usr/bin/env python
"""Reconstruct one CNM/APS 26-ID-C Hard X-ray Nanoprobe ptychography dataset through the ptychodus api.

26-ID-C has written three detector layouts beside one EPICS MDA scan record: Medipix3
frames as one gzipped TIFF per point (2019) or one plain TIFF per point (2022), and an
Eiger stack in one HDF5 file beside an instrument-snapshot member (2023). The diffraction
reader is chosen by the suffix of ``--diffraction-file``; the positions are the
zone-plate piezo encoders the MDA records as detector channels in every era.

Scope and limitations
---------------------

- One scan, stock LSQML unless ``--ptychi-options`` names something else.
- The initial probe is the back-propagated mean pattern unless ``--fzp-preset`` names a
  zone plate or ``--probe-file`` a warm start; ``--probe-propagation-distance-m`` moves
  it off focus. The zone plate's defocus is recorded in no file, so the data-derived
  probe asserts less.
- The energy and detector distance come from the MDA, or from the 2023 snapshot, which
  takes precedence. Some MDA records carry neither; pass ``--photon-energy-eV`` and
  ``--detector-distance-m`` for those. The pixel pitch is the detector's, supplied by
  the reader.
- Scans taken with the detector arm off the forward direction (Bragg or Laue) need
  ``--two-theta-deg``, ``--detector-gamma-deg`` and ``--incidence-angle-deg``: the
  positions are then carried into the exit frame, where the transmission forward model
  holds for a thin sample. The reader does not record these angles, and without them
  such scans are reconstructed as transmission, which is not physically meaningful.
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
    logger_name='reconstruct_26idc',
    description=(
        'CNM/APS 26-ID-C Hard X-ray Nanoprobe ptychography reconstruction via the ptychodus api.'
    ),
    product_name='26idc-reconstruct',
    diffraction_reader='CNM_APS_HXN_TIFF',
    position_reader='CNM_APS_HXN',
    diffraction_file_help=(
        'Any member of the scan: scan_<N>_img_<F>.tif or .tif.gz (the rest of the series '
        'is globbed), or scan_<N>_<F>.h5, including the scan_<N>_000000.h5 snapshot.'
    ),
    position_file_help=(
        'The EPICS MDA scan record, 26idbSOFT_<NNNN>.mda; the reader also finds it beside '
        'the frames for the metadata.'
    ),
    fzp_preset_help=(
        'FresnelZonePlate plugin preset for a model probe, e.g. "CNM/APS 26-ID-C Hard '
        'X-ray Nanoprobe". Empty by default, so the probe is estimated from the mean '
        'pattern instead. Ignored with --probe-file.'
    ),
    diffraction_readers_by_suffix={
        '.gz': 'CNM_APS_HXN_TIFF',
        '.tif': 'CNM_APS_HXN_TIFF',
        '.tiff': 'CNM_APS_HXN_TIFF',
        '.h5': 'CNM_APS_HXN',
        '.hdf5': 'CNM_APS_HXN',
    },
    offers_scattering_geometry=True,
)


def main() -> ExitCode:
    return run_standard_reconstruction(PROFILE)


if __name__ == '__main__':
    sys.exit(main())

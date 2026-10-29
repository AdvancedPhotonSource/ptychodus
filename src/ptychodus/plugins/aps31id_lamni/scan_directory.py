"""Locating one scan's detector directory in an APS 31-ID-E experiment.

LamNI writes each scan's detector series to ``eiger_4/<block>/S<NNNNN>/``, where ``<block>``
is named ``S<start>-<end>`` for the run of scan numbers it holds. How many scans a block
holds has varied between experiments -- a thousand in some, a hundred in others -- so the
block is not derivable from the scan number alone.
"""

from __future__ import annotations

from pathlib import Path


def locate_scan_directory(detector_dir: Path, scan_no: int) -> Path:
    """Find ``<detector_dir>/<block>/S<NNNNN>`` without assuming how many scans a block holds.

    The scan directory is found by name under any block rather than computed from a block
    size. A scan found under more than one block is an error: picking one would reconstruct
    an unpredictable copy.
    """
    name = f'S{scan_no:05d}'
    candidates = sorted(path for path in detector_dir.glob(f'S*-*/{name}') if path.is_dir())

    if not candidates:
        raise FileNotFoundError(f'No directory for scan {scan_no} under {detector_dir}.')

    if len(candidates) > 1:
        blocks = ', '.join(path.parent.name for path in candidates)
        raise FileNotFoundError(
            f'Scan {scan_no} appears under several blocks of {detector_dir}: {blocks}.'
        )

    return candidates[0]

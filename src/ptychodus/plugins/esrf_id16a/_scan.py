"""Scan-layout resolution and header parsing shared by the ESRF ID16A readers.

An ID16A near-field ptychography scan directory holds a metadata master written by
SPEC, one or more LIMA frame files, and a dark stack. Either the master or any frame
file is a valid entry point, so these helpers resolve the rest of the scan from
whichever one the caller opened.

Three layout details are easy to guess wrong and are handled here rather than in each
reader:

- The master's entry group is named ``entry_0000: <title>`` -- the title is part of
  the key, so the group cannot be indexed by a literal ``entry_0000``.
- ``definition`` is a dataset under that group holding ``NFP``, not a file attribute.
- A LIMA file's entry group *is* plain ``entry_0000``, and its detector group carries
  the detector-interface name rather than the camera name.

Header entries are written by SPEC with inconsistent padding: some keys carry a
trailing space and every value is space-padded, so both are stripped on the way in.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Final
import logging
import re

import h5py

logger = logging.getLogger(__name__)

_DEFINITION: Final[str] = 'NFP'
_LIMA_CREATOR_PREFIX: Final[str] = 'LIMA'
_LIMA_ENTRY: Final[str] = 'entry_0000'
_DARK_STEM: Final[str] = 'dark_0000'

# A LIMA frame file is the scan title followed by a four-digit series number.
_SERIES_SUFFIX: Final[re.Pattern[str]] = re.compile(r'(\d{4})$')


def _decode(value: object) -> str:
    """Return an HDF5 scalar as a stripped string, whatever width or encoding it has."""
    if isinstance(value, bytes):
        return value.decode('utf-8', errors='replace').strip()

    return str(value).strip()


def is_master_file(h5_file: h5py.File) -> bool:
    """Whether this file is the SPEC-written metadata master of an NFP scan."""
    entry = master_entry(h5_file)

    if entry is None:
        return False

    definition = entry.get('definition')
    return isinstance(definition, h5py.Dataset) and _decode(definition[()]) == _DEFINITION


def master_entry(h5_file: h5py.File) -> h5py.Group | None:
    """The master's single entry group, whose name embeds the scan title."""
    for name in h5_file:
        if name.startswith(_LIMA_ENTRY):
            group = h5_file[name]

            if isinstance(group, h5py.Group):
                return group

    return None


def is_lima_file(h5_file: h5py.File) -> bool:
    """Whether this file was written by LIMA, i.e. holds detector frames."""
    return _decode(h5_file.attrs.get('creator', '')).startswith(_LIMA_CREATOR_PREFIX)


def detector_group(h5_file: h5py.File) -> h5py.Group:
    """The LIMA detector group, found by looking for the one that holds ``data``.

    The name is a detector-interface label rather than the camera name, so it is
    discovered rather than assumed: another ID16A configuration may present a
    different interface with the same data underneath.
    """
    instrument = h5_file.get(f'{_LIMA_ENTRY}/instrument')

    if isinstance(instrument, h5py.Group):
        for name in instrument:
            group = instrument[name]

            if isinstance(group, h5py.Group) and isinstance(group.get('data'), h5py.Dataset):
                return group

    raise ValueError(
        f'"{h5_file.filename}" has no {_LIMA_ENTRY}/instrument group holding a "data" dataset.'
    )


def detector_data_path(h5_file: h5py.File) -> str:
    """Path to the frame stack, for a reader that opens the file lazily later."""
    return f'{detector_group(h5_file).name}/data'


def read_header(h5_file: h5py.File) -> Mapping[str, str]:
    """The LIMA header collection, with keys and values both stripped.

    SPEC writes a trailing space into some keys (``'pixel_size '``, ``'energy '``,
    ``'optic_used '``) and not others (``'propagation_distance'``), and pads every
    value, so neither can be used as stored.
    """
    header = detector_group(h5_file).get('header')

    if not isinstance(header, h5py.Group):
        raise ValueError(f'"{h5_file.filename}" has no detector header collection.')

    return {str(key).strip(): _decode(header[key][()]) for key in header}


def scan_title(file_path: Path) -> str:
    """The scan title, which names both the directory contents and the series files.

    Taken from the master when the entry point is one, and otherwise from the frame
    file's own name with its series number removed.
    """
    with h5py.File(file_path, 'r') as h5_file:
        if is_master_file(h5_file):
            entry = master_entry(h5_file)

            if entry is not None:
                title = entry.get('title')

                if isinstance(title, h5py.Dataset):
                    return _decode(title[()])

    return _SERIES_SUFFIX.sub('', file_path.stem)


def series_files(file_path: Path) -> list[Path]:
    """Every LIMA frame file of this scan, ordered by series number.

    Siblings are resolved by basename within the scan directory. The master records
    absolute beamline paths in ``measurement/files``, which do not survive a copy off
    the facility filesystem and are deliberately not consulted.
    """
    directory = file_path.parent
    title = scan_title(file_path)
    candidates: list[tuple[int, Path]] = []

    for candidate in directory.glob(f'{title}[0-9][0-9][0-9][0-9]{file_path.suffix}'):
        match = _SERIES_SUFFIX.search(candidate.stem)

        if match is not None:
            candidates.append((int(match.group(1)), candidate))

    if not candidates:
        raise ValueError(f'No ESRF ID16A frame files for "{title}" beside "{file_path}".')

    return [path for _, path in sorted(candidates)]


def dark_file(file_path: Path) -> Path | None:
    """The dark stack beside this scan, when the directory holds one."""
    candidate = file_path.parent / f'{_DARK_STEM}{file_path.suffix}'
    return candidate if candidate.is_file() else None


def master_file(file_path: Path) -> Path | None:
    """The scan's metadata master, whatever the entry point was.

    The master's name encodes the proposal and sample rather than the scan title, so
    it cannot be derived from the title the way the frame files can. The directory
    holds few enough files to identify it by opening the candidates instead.
    """
    for candidate in sorted(file_path.parent.glob(f'*{file_path.suffix}')):
        try:
            with h5py.File(candidate, 'r') as h5_file:
                if is_master_file(h5_file):
                    return candidate
        except OSError:
            continue

    return None


def read_sample_x0_mm(file_path: Path) -> float | None:
    """The master's reference sample position, which anchors the geometry cross-check."""
    master = master_file(file_path)

    if master is None:
        return None

    with h5py.File(master, 'r') as h5_file:
        entry = master_entry(h5_file)

        if entry is None:
            return None

        sample_x0 = entry.get('TOMO/sx0')

        if not isinstance(sample_x0, h5py.Dataset):
            return None

        return float(sample_x0[()])


def parse_motor(header: Mapping[str, str], name: str) -> float | None:
    """One motor position by name, from the parallel ``motor_mne`` / ``motor_pos`` lists.

    The lists are **not** uniform in units -- ``sx`` is in millimeters while ``spy``
    and ``spz`` are in micrometers -- so a caller asks for one motor and applies the
    unit that motor is recorded in, rather than scaling the list as a whole.
    """
    names = header.get('motor_mne', '').split()
    values = header.get('motor_pos', '').split()

    if len(names) != len(values):
        logger.warning('motor_mne and motor_pos disagree in length; ignoring both.')
        return None

    try:
        return float(values[names.index(name)])
    except (ValueError, IndexError):
        return None


def parse_float_list(header: Mapping[str, str], key: str, expected: int) -> list[float]:
    """A whitespace-separated numeric header entry, length-checked against the frames."""
    try:
        values = [float(token) for token in header[key].split()]
    except KeyError:
        raise ValueError(f'The detector header has no "{key}" entry.') from None
    except ValueError as exc:
        raise ValueError(f'The "{key}" header entry is not all numeric: {exc}') from None

    if len(values) != expected:
        raise ValueError(
            f'The "{key}" header entry holds {len(values)} values but the file has '
            f'{expected} frames.'
        )

    return values

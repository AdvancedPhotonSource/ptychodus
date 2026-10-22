"""Synthetic ESRF ID16A scan files, shaped like the real ones.

The layout quirks these fixtures reproduce are the ones a reader gets wrong by
guessing: the master's entry group embeds the scan title in its key, ``definition``
is a dataset rather than an attribute, and SPEC writes a trailing space into some
header keys and pads every value. Numbers are the real 55 nm geometry so the derived
distances can be checked against values measured off the instrument.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy

# The real 55 nm scan, which the geometry assertions are written against.
OPTIC_USED_UM = 3.03751
PIXEL_SIZE_UM = 0.0549996
PROPAGATION_DISTANCE_MM = 21.477
ENERGY_KEV = 17.1
SAMPLE_X_MM = 17.873
SAMPLE_X0_MM = -4.0

DETECTOR_EXTENT_PX = 8
FRAMES_PER_FILE = 4
DARK_FRAMES = 5
DARK_LEVEL = 100


def _write_header(
    detector: h5py.Group, *, spy: list[float], spz: list[float], num_frames: int
) -> None:
    """The LIMA header collection, padded exactly the way SPEC pads it."""
    header = detector.create_group('header')

    # Trailing space on some keys and not others, and a leading space on every value:
    # both are how the real files are written, and both must survive stripping.
    header['optic_used '] = numpy.bytes_(f' {OPTIC_USED_UM}')
    header['pixel_size '] = numpy.bytes_(f' {PIXEL_SIZE_UM}')
    header['energy '] = numpy.bytes_(f' {ENERGY_KEV}')
    header['propagation_distance'] = numpy.bytes_(f'{PROPAGATION_DISTANCE_MM}')
    header['motor_mne '] = numpy.bytes_(' dummy somega sx sy sz spy spz')
    header['motor_pos '] = numpy.bytes_(f' 0.001 52 {SAMPLE_X_MM} -1.46 1.48 {spy[0]} {spz[0]}')
    header['acq_nb_frames'] = numpy.bytes_(f'{num_frames}')
    header['image_flip'] = numpy.bytes_('<flip x : False,flip y : True>')
    header['camera_type '] = numpy.bytes_(' ximea')
    header['spy'] = numpy.bytes_(' '.join(f'{value}' for value in spy))
    header['spz'] = numpy.bytes_(' '.join(f'{value}' for value in spz))


def write_lima_file(
    file_path: Path,
    *,
    spy: list[float],
    spz: list[float],
    fill: int = 1000,
    detector_name: str = 'PCIe',
) -> None:
    """One LIMA frame file: patterns, the acquisition collection and the header."""
    num_frames = len(spy)

    with h5py.File(file_path, 'w') as h5_file:
        h5_file.attrs['creator'] = 'LIMA-1.9.12'
        entry = h5_file.create_group('entry_0000')
        detector = entry.create_group(f'instrument/{detector_name}')
        detector.create_dataset(
            'data',
            data=numpy.full(
                (num_frames, DETECTOR_EXTENT_PX, DETECTOR_EXTENT_PX), fill, dtype=numpy.uint16
            ),
        )
        acquisition = detector.create_group('acquisition')
        acquisition['exposure_time'] = 0.2
        _write_header(detector, spy=spy, spz=spz, num_frames=num_frames)


def write_master_file(file_path: Path, *, title: str) -> None:
    """The SPEC master, whose entry group name carries the scan title."""
    with h5py.File(file_path, 'w') as h5_file:
        # The title is part of the group key, not a sibling dataset only.
        entry = h5_file.create_group(f'entry_0000: SAMPLE - {title}')
        entry['definition'] = numpy.bytes_('NFP')
        entry['dataset_type'] = numpy.bytes_('NFP_image')
        entry['title'] = numpy.bytes_(title)
        entry['TOMO/sx0'] = SAMPLE_X0_MM
        entry['PTYCHO/propagation'] = numpy.bytes_('nearfield')


def write_dark_file(file_path: Path, *, hot_pixel: tuple[int, int] | None = (1, 2)) -> None:
    """A dark stack at a flat pedestal, optionally with one hot and one stuck pixel."""
    stack = numpy.full(
        (DARK_FRAMES, DETECTOR_EXTENT_PX, DETECTOR_EXTENT_PX), DARK_LEVEL, dtype=numpy.uint16
    )

    if hot_pixel is not None:
        stack[:, hot_pixel[0], hot_pixel[1]] = DARK_LEVEL * 20
        # Saturated in one frame only, which is enough to disqualify the pixel.
        stack[0, 0, 0] = numpy.iinfo(numpy.uint16).max

    with h5py.File(file_path, 'w') as h5_file:
        h5_file.attrs['creator'] = 'LIMA-1.9.12'
        detector = h5_file.create_group('entry_0000/instrument/PCIe')
        detector.create_dataset('data', data=stack)


def write_scan(
    directory: Path, *, title: str = 'sample_055nm_nfp01', num_files: int = 1, reverse: bool = False
) -> Path:
    """A whole scan directory. Returns the master's path.

    `reverse` traverses the same positions in the opposite order, which is what the
    real ``nfp02`` repeat does and what makes index pairing worth testing.
    """
    directory.mkdir(parents=True, exist_ok=True)
    master = directory / f'md1404-SAMPLE-{title}.h5'
    write_master_file(master, title=title)
    write_dark_file(directory / 'dark_0000.h5')

    total = num_files * FRAMES_PER_FILE
    spy_all = [round(-5.0 + index, 3) for index in range(total)]
    spz_all = [round(+5.0 - index, 3) for index in range(total)]

    if reverse:
        spy_all = spy_all[::-1]
        spz_all = spz_all[::-1]

    for series in range(num_files):
        start = series * FRAMES_PER_FILE
        stop = start + FRAMES_PER_FILE
        write_lima_file(
            directory / f'{title}{series:04d}.h5',
            spy=spy_all[start:stop],
            spz=spz_all[start:stop],
        )

    return master

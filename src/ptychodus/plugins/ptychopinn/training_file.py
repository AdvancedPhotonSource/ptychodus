"""The NPZ archive PtychoPINN trains from: written from a ReconstructInput, found, merged.

Pure numpy: nothing here imports the backend, so the format is on offer whether or
not PtychoPINN is installed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any
import logging
import zipfile

import numpy

from ptychodus.api.diffraction import BadPixels, DiffractionPatterns
from ptychodus.api.files import FileFormat
from ptychodus.api.preprocess import diffraction as preprocess
from ptychodus.api.reconstruct import ReconstructInput, TrainingDataFileWriter
from ptychodus.api.typing import AnyArrayType, ComplexArrayType

__all__ = [
    'PtychoPINNTrainingDataFileWriter',
    'find_ptychopinn_training_archives',
    'merge_ptychopinn_training_archives',
    'save_ptychopinn_training_data',
]

logger = logging.getLogger(__name__)


def _repair_bad_pixels(
    parameters: ReconstructInput, *, inpaint: bool
) -> tuple[DiffractionPatterns, BadPixels]:
    """Repair the patterns, and return them with the mask saying which pixels were filled.

    An absent mask becomes an all-False one, so the archive always has a mask to store.
    """
    patterns = parameters.diffraction_patterns
    bad_pixels = parameters.bad_pixels

    if bad_pixels is None:
        bad_pixels = numpy.zeros(patterns.shape[1:], dtype=bool)

    repaired = (
        preprocess.inpaint_bad_pixels(patterns, bad_pixels)
        if inpaint
        else preprocess.zero_bad_pixels(patterns, bad_pixels)
    )

    return repaired, bad_pixels


def save_ptychopinn_training_data(
    file_path: Path,
    parameters: ReconstructInput,
    *,
    inpaint_bad_pixels: bool = True,
) -> None:
    """Write a ReconstructInput to the NPZ format consumed by PtychoPINN trainers.

    Bad pixels are filled by biharmonic inpainting, or zeroed when
    `inpaint_bad_pixels` is False; `diff3d` keeps the source dtype either way, and
    `bad_pixels` records which positions were filled rather than measured.
    `probeGuess` is the full `(N_modes, H, W)` mode stack at OPR index zero. All
    scan points are assigned `scan_index=0` (single-object assumption).

    The archive declares its measurement domain as photon counts, which is what
    a detector reports and what this function passes through unscaled. A reader
    compares that declaration against the scaling its model was trained under
    and converts or rejects accordingly, so an undeclared archive would instead
    be guessed at. Revisit the declaration if a preprocessing step is ever
    added that normalizes or square-roots the patterns, because it would make
    the pair below untrue rather than merely imprecise.
    """
    object_geometry = parameters.product.object_.get_geometry()
    positions = parameters.product.probe_positions
    xcoords = object_geometry.map_probe_positions_to_object_x_px(positions)
    ycoords = object_geometry.map_probe_positions_to_object_y_px(positions)
    diff3d, bad_pixels = _repair_bad_pixels(parameters, inpaint=inpaint_bad_pixels)

    numpy.savez(
        file_path,
        xcoords=xcoords,
        ycoords=ycoords,
        xcoords_start=xcoords,
        ycoords_start=ycoords,
        diff3d=diff3d,
        bad_pixels=bad_pixels,
        probeGuess=parameters.product.probes.get_mean_probe().get_array(),
        objectGuess=parameters.product.object_.get_layer(0),
        scan_index=numpy.zeros(len(parameters.product.probe_positions), dtype=int),
        scale_contract_version='ci_intensity_v2',
        measurement_domain='count_intensity',
    )


def find_ptychopinn_training_archives(directory: Path) -> list[Path]:
    """The training archives directly in `directory`, sorted by name.

    An archive is any ``*.npz`` file holding the coordinates, patterns and probe a
    training run needs; subdirectories are not searched. A file that is not one is
    skipped with a warning saying why, so an unrelated array beside the training data
    costs a log line rather than the run.

    Raises:
        FileNotFoundError: if no training archive is found.
    """
    required_keys = ('xcoords', 'ycoords', 'diff3d', 'probeGuess')
    archives: list[Path] = []

    for file_path in sorted(directory.glob('*.npz')):
        if not file_path.is_file():
            continue

        try:
            with numpy.load(file_path, allow_pickle=False) as archive:
                missing = [key for key in required_keys if key not in archive.files]
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            logger.warning(f'Skipping {file_path.name}: not a readable NPZ archive ({exc}).')
            continue

        if missing:
            logger.warning(f'Skipping {file_path.name}: no {", ".join(missing)}.')
        else:
            archives.append(file_path)

    if not archives:
        raise FileNotFoundError(f'No PtychoPINN training archives in {directory}.')

    return archives


def merge_ptychopinn_training_archives(
    file_paths: Sequence[Path],
    destination: Path,
    *,
    truth_patches: Callable[[Path], ComplexArrayType] | None = None,
) -> Path:
    """Write the archives as one object bank, the form PtychoPINN trains several objects from.

    Each archive is one object. Their frames are concatenated in the order given, with
    ``object_index`` set to each archive's position so that neighbor groups never mix
    objects, and ``scan_index`` carried through unchanged. One ``objectGuess`` cannot hold
    several objects, so the bank carries none; `truth_patches`, when given, supplies each
    archive's ``(M, N, N)`` ground-truth patches instead, stored as ``Y``. The ptychodus-only
    ``bad_pixels`` is dropped, since the patterns were repaired when written.

    Returns:
        The path written, `destination` with an ``.npz`` suffix.

    Raises:
        ValueError: if the archives disagree on pattern shape, probe or scale contract,
            or the patches do not match the patterns. The bank stores one probe, so data
            measured with different probes must be trained from separate directories.
    """
    frame_keys = ('diff3d', 'xcoords', 'ycoords', 'xcoords_start', 'ycoords_start', 'scan_index')
    contract_keys = ('scale_contract_version', 'measurement_domain')
    frames: dict[str, list[AnyArrayType]] = {key: [] for key in (*frame_keys, 'object_index')}
    patches: list[ComplexArrayType] = []
    first_path = file_paths[0]
    probe: ComplexArrayType | None = None
    contract: tuple[str | None, ...] = ()

    for object_index, file_path in enumerate(file_paths):
        with numpy.load(file_path, allow_pickle=False) as archive:
            values = {key: archive[key] for key in archive.files}

        num_frames = len(values['xcoords'])
        values.setdefault('xcoords_start', values['xcoords'])
        values.setdefault('ycoords_start', values['ycoords'])
        values.setdefault('scan_index', numpy.zeros(num_frames, dtype=numpy.int64))
        file_contract = tuple(str(values[key]) if key in values else None for key in contract_keys)

        if probe is None:
            probe = values['probeGuess']
            contract = file_contract
        elif values['diff3d'].shape[1:] != frames['diff3d'][0].shape[1:]:
            raise ValueError(
                f'{file_path.name} has {values["diff3d"].shape[1:]} patterns, '
                f'but {first_path.name} has {frames["diff3d"][0].shape[1:]}.'
            )
        elif values['probeGuess'].shape != probe.shape or not numpy.allclose(
            values['probeGuess'], probe
        ):
            raise ValueError(
                f'{file_path.name} and {first_path.name} hold different probes. A training '
                'directory holds data sharing one probe; train other probes separately.'
            )
        elif file_contract != contract:
            raise ValueError(
                f'{file_path.name} declares scale contract {file_contract}, '
                f'but {first_path.name} declares {contract}.'
            )

        for key in frame_keys:
            frames[key].append(values[key])

        frames['object_index'].append(numpy.full(num_frames, object_index, dtype=numpy.int64))

        if truth_patches is not None:
            file_patches = truth_patches(file_path)

            if file_patches.shape != values['diff3d'].shape:
                raise ValueError(
                    f'Ground truth for {file_path.name} has shape {file_patches.shape}, '
                    f'not the pattern shape {values["diff3d"].shape}.'
                )

            patches.append(file_patches)

    bank: dict[str, Any] = {key: numpy.concatenate(arrays) for key, arrays in frames.items()}
    bank['probeGuess'] = numpy.asarray(probe)

    if patches:
        bank['Y'] = numpy.concatenate(patches)

    for key, value in zip(contract_keys, contract):
        if value is not None:
            bank[key] = numpy.asarray(value)

    file_path = destination.with_suffix('.npz')
    numpy.savez(file_path, **bank)
    return file_path


class PtychoPINNTrainingDataFileWriter(TrainingDataFileWriter):
    """Writes the NPZ archive PtychoPINN trains from; see `save_ptychopinn_training_data`."""

    def __init__(self, *, inpaint_bad_pixels: bool = True) -> None:
        self._inpaint_bad_pixels = inpaint_bad_pixels

    def get_file_description(self) -> str:
        return 'PtychoPINN Training Data'

    def get_file_extensions(self) -> set[str]:
        return FileFormat.NPZ.extensions

    def write(self, file_path: Path, parameters: ReconstructInput) -> None:
        save_ptychopinn_training_data(
            file_path, parameters, inpaint_bad_pixels=self._inpaint_bad_pixels
        )

"""The NPZ archive PtychoPINN trains from, written from a ReconstructInput.

Pure numpy: nothing here imports the backend, so the format is on offer whether or
not PtychoPINN is installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy

from ptychodus.api.diffraction import BadPixels, DiffractionPatterns
from ptychodus.api.files import FileFormat
from ptychodus.api.preprocess import diffraction as preprocess
from ptychodus.api.reconstruct import ReconstructInput, TrainingDataFileWriter

__all__ = [
    'PtychoPINNTrainingDataFileWriter',
    'save_ptychopinn_training_data',
]


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
        probeGuess=parameters.product.probes.get_probe_no_opr().get_array(),
        objectGuess=parameters.product.object_.get_layer(0),
        scan_index=numpy.zeros(len(parameters.product.probe_positions), dtype=int),
        scale_contract_version='ci_intensity_v2',
        measurement_domain='count_intensity',
    )


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

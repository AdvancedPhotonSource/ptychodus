"""The filename convention that pairs a fold_slice scan's two files.

The preprocessing step writes each scan as ``<stem>_dp.hdf5`` holding the patterns and
``<stem>_para.hdf5`` holding the probe positions and the experiment geometry. Neither
file refers to the other, so the only thing that pairs them is the suffix, and that
makes it a property of the format rather than of whoever is reading it.
"""

from pathlib import Path

_DIFFRACTION_SUFFIX = '_dp'
_POSITION_SUFFIX = '_para'


def find_position_file(diffraction_file: Path) -> Path | None:
    """The ``_para`` companion of a ``_dp`` pattern file, or None.

    Returns None for a name off the convention, which is the signal that the position
    file has to be named explicitly rather than derived. The returned path is not
    checked for existence: a caller that can fall back wants to distinguish "this name
    says nothing" from "the companion is missing".
    """
    stem = diffraction_file.stem

    if not stem.endswith(_DIFFRACTION_SUFFIX):
        return None

    base = stem[: -len(_DIFFRACTION_SUFFIX)]
    return diffraction_file.with_name(f'{base}{_POSITION_SUFFIX}{diffraction_file.suffix}')

"""Unit tests for ptychodus.api.files.FileHandler.

The base contributes exactly one piece of behavior: composing a Qt name filter from
a description and a set of extensions. These tests pin that composition, including
the sort that keeps the filter -- and so the name a plugin registers under -- stable
however the extension set happens to iterate.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ptychodus.api.files import FileHandler, FileReader, FileWriter


class _Handler(FileHandler):
    def __init__(self, description: str, extensions: set[str]) -> None:
        self._description = description
        self._extensions = extensions

    def get_file_description(self) -> str:
        return self._description

    def get_file_extensions(self) -> set[str]:
        return self._extensions


def test_single_extension() -> None:
    handler = _Handler('NumPy Binary Files', {'.npy'})
    assert handler.get_file_filter() == 'NumPy Binary Files (*.npy)'


def test_multiple_extensions_are_sorted() -> None:
    handler = _Handler('Tagged Image File Format Files', {'.tiff', '.tif'})
    assert handler.get_file_filter() == 'Tagged Image File Format Files (*.tif *.tiff)'


def test_filter_is_independent_of_set_iteration_order() -> None:
    forward = _Handler('Ptychodus Product Files', {'.h5', '.hdf5'})
    reverse = _Handler('Ptychodus Product Files', {'.hdf5', '.h5'})
    assert forward.get_file_filter() == reverse.get_file_filter()
    assert forward.get_file_filter() == 'Ptychodus Product Files (*.h5 *.hdf5)'


@pytest.mark.parametrize(
    'description, extensions, expected',
    [
        ('APS 4-ID-B,G,H POLAR Files', {'.hdf'}, 'APS 4-ID-B,G,H POLAR Files (*.hdf)'),
        (
            'APS 33-ID-C VelociProbe Files - Laser Interferometer',
            {'.txt'},
            'APS 33-ID-C VelociProbe Files - Laser Interferometer (*.txt)',
        ),
        (
            'Coherent X-ray Imaging Files, SHARP layout',
            {'.cxi'},
            'Coherent X-ray Imaging Files, SHARP layout (*.cxi)',
        ),
    ],
)
def test_punctuation_in_description_is_preserved(
    description: str, extensions: set[str], expected: str
) -> None:
    assert _Handler(description, extensions).get_file_filter() == expected


def test_a_reader_must_still_describe_itself() -> None:
    """The format contract is not waived by implementing only the payload half."""

    class _Reader(FileReader[int]):
        def get_file_extensions(self) -> set[str]:
            return {'.npy'}

        def read(self, file_path: Path) -> int:
            return 0

    with pytest.raises(TypeError, match='get_file_description'):
        _Reader()  # type: ignore[abstract]


def test_a_writer_must_still_describe_itself() -> None:
    class _Writer(FileWriter[int]):
        def get_file_extensions(self) -> set[str]:
            return {'.npy'}

        def write(self, file_path: Path, value: int) -> None:
            pass

    with pytest.raises(TypeError, match='get_file_description'):
        _Writer()  # type: ignore[abstract]
